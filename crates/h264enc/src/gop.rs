//! GOP-parallel encoding (feature `threads`) for the rate controls whose frame QP needs no feedback
//! from earlier frames: constant QP and constant quality.
//!
//! `try_encode` keeps its serial input side: import, half-resolution analysis against the previous
//! source picture, AQ offsets, the IDR / scene-cut decision and the mini-GOP structure run on the
//! calling thread in display order, exactly as on the serial path, and every picture gets its
//! coding-order index and decoding time there. Only the coding moves. Each GOP gets a thread with
//! its own reference state, started at its IDR; the batches of the GOP (its IDR, then per mini-GOP
//! the P picture and its B pictures, then whatever the next IDR or a flush finishes) go to that
//! thread as they are formed, which codes them in that order with the same `encode_batch` calls as
//! the serial path. The next IDR closes the GOP and starts the next thread, so a GOP that falls
//! behind keeps coding while the next ones start: GOPs run concurrently as soon as the input comes
//! faster than one GOP at a time can code it. Slices still run on the encoder's pool, which all GOP
//! threads share.
//!
//! Closed GOPs make this exact: an IDR empties the DPB and restarts `frame_num` and the POC, so a
//! GOP reads nothing of the GOPs before it but a few stream counters, which its thread starts from:
//! the `idr_pic_id` parity, the reference picture numbering and whether the identification SEI went
//! out (with the first picture of the stream only). Constant QP and constant quality pick a frame's
//! QP from its type alone; the bitrate modes feed each picture's size into the next QP and stay
//! serial.
//!
//! Results come back in decoding order: a GOP's coded pictures (packets, rate-control statistics,
//! reconstructions) are taken (`Encoder::deliver`) as they are coded once every GOP before it is
//! done, so a packet never comes out before one that precedes it. Memory is bounded by
//! [`GOP_BUFFER_BYTES`]. A GOP thread that panics becomes [`Error::Worker`] for that GOP only.
//! Dropping the encoder stops every GOP thread after its current batch and waits for it, so the
//! GOPs' pictures, buffers and pool work are released by the time the drop returns.

use std::any::Any;
use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread::JoinHandle;

use crate::nal::SliceType;
use crate::ratecontrol::FixedQp;
use crate::{Batch, Coded, Ctx, Encoder, EncoderConfig, Error, Refs, encode_batch};

/// Bytes of padded pictures the GOP-parallel path may hold: the input pictures not coded yet plus
/// the working set of every GOP thread. 1 GiB holds about four 2-second GOPs of 1080p30 (60
/// pictures of 3.6 MiB each). Two GOPs of the longest kind (`keyint` pictures) must fit or the
/// encoder stays serial: 4K (13 MiB a picture) with a 2-second `keyint` does.
pub(crate) const GOP_BUFFER_BYTES: usize = 1 << 30;

/// Spare pictures the calling thread may keep besides its inputs (its frame pool).
const SPARE_FRAMES: usize = 8;

/// Stack of a GOP thread: slices usually run on the pool, a single-slice picture runs here.
const GOP_STACK: usize = 8 << 20;

/// Coded pictures of one batch, each with the reconstruction capture generation of the batch.
type Out = Vec<(Coded, Option<u64>)>;

/// The GOPs of a GOP-parallel encoder.
pub(crate) struct Gops {
    fixed: FixedQp,
    /// GOPs not taken completely yet, oldest first. The last one is open (it receives the batches)
    /// unless a flush closed it.
    gops: VecDeque<Gop>,
    /// Number of the next GOP.
    next_seq: u64,
    /// IDR and reference (I/P) pictures handed to GOPs so far: a GOP's counters start there.
    idrs: u32,
    refs: u32,
    /// [`GOP_BUFFER_BYTES`] in padded pictures.
    budget: usize,
    /// Padded pictures a GOP thread needs besides its inputs: references with their three
    /// half-sample planes (about three pictures each), the pictures being reconstructed, spares.
    work: usize,
    /// GOP threads at once.
    max_gops: usize,
    /// The pool that analyses the input pictures (see `Encoder::analyse`); `None` if it could not
    /// be built (the encoder's pool then does it).
    analysis: Option<rayon::ThreadPool>,
    /// Set when the encoder goes away: GOP threads stop after their current batch.
    cancel: Arc<AtomicBool>,
    /// Make the GOP with this number panic on its thread.
    #[cfg(test)]
    pub(crate) plant_panic: Option<u64>,
    /// Make every GOP thread hold its batches until the encoder goes away, then take a while over
    /// the one it is coding, as a GOP busy with a long batch would.
    #[cfg(test)]
    plant_hold: bool,
    /// GOP threads started and not ended yet.
    #[cfg(test)]
    live: Arc<AtomicUsize>,
}

/// The test hooks a GOP thread starts with.
#[cfg(test)]
struct Plant {
    /// Panic on the thread.
    panic: bool,
    /// See [`Gops::plant_hold`].
    hold: bool,
    /// Counts the thread in [`Gops::live`] until it ends.
    alive: Alive,
}

/// One GOP thread counted in [`Gops::live`], from before it is started until it ends.
#[cfg(test)]
struct Alive(Arc<AtomicUsize>);

#[cfg(test)]
impl Alive {
    /// Count one more thread in `live`.
    fn new(live: &Arc<AtomicUsize>) -> Alive {
        live.fetch_add(1, Ordering::SeqCst);
        Alive(Arc::clone(live))
    }
}

#[cfg(test)]
impl Drop for Alive {
    /// The thread ended (or was never started).
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// One GOP (or the rest of one, after a flush) and what codes it.
struct Gop {
    /// Still receiving batches.
    open: bool,
    /// Input pictures sent to the coder, and taken back coded.
    sent: usize,
    taken: usize,
    /// Pictures coded so far (their input pictures are free).
    coded: Arc<AtomicUsize>,
    /// The coder failed while the GOP was open: its batches are dropped, and the failure is
    /// reported once the GOP is closed and taken.
    dead: bool,
    coder: Coder,
}

enum Coder {
    Thread {
        /// Batches to code; dropped to close the GOP.
        batches: Option<mpsc::Sender<Batch>>,
        coded: mpsc::Receiver<Out>,
        /// The reference state the GOP ended with, or the panic.
        handle: Option<JoinHandle<Result<Refs, String>>>,
    },
    /// No thread could be started: the GOP is coded on the calling thread as its batches come.
    Here { refs: Refs, done: VecDeque<Out>, failed: Option<String> },
}

/// What [`Gop::next`] found.
enum Next {
    Coded(Out),
    /// Nothing yet.
    Pending,
    /// Every coded picture was taken and the coder is done.
    Finished(Result<Refs, String>),
}

impl Gop {
    /// Start coding a GOP from `refs` on a thread (or here, when no thread can be had).
    fn start(ctx: &Arc<Ctx>, fixed: FixedQp, refs: Refs, cancel: &Arc<AtomicBool>, #[cfg(test)] plant: Plant) -> Gop {
        let coded = Arc::new(AtomicUsize::new(0));
        let (batch_tx, batch_rx) = mpsc::channel::<Batch>();
        let (out_tx, out_rx) = mpsc::channel::<Out>();
        // the reference state goes over a channel so it stays here if the thread cannot be started
        let (refs_tx, refs_rx) = mpsc::sync_channel::<Refs>(1);
        let (c, count, stop) = (Arc::clone(ctx), Arc::clone(&coded), Arc::clone(cancel));
        let body = move || -> Result<Refs, String> {
            #[cfg(test)]
            let Plant { panic: plant_panic, hold, alive: _alive } = plant;
            let refs = refs_rx.recv().map_err(|_| "the GOP never reached its thread".to_string())?;
            catch_unwind(AssertUnwindSafe(|| {
                #[cfg(test)]
                if hold {
                    while !stop.load(Ordering::Relaxed) {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                    // the rest of a long batch
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
                #[cfg(test)]
                if plant_panic {
                    // from a pool task, as a panic in a slice would come
                    let boom = || -> u8 { panic!("planted GOP panic") };
                    c.pool.install(|| rayon::join(|| 0u8, boom));
                }
                code(&c, fixed, refs, &batch_rx, &out_tx, &count, &stop)
            }))
            .map_err(panic_text)
        };
        let spawned = std::thread::Builder::new().name("h264enc-gop".into()).stack_size(GOP_STACK).spawn(body);
        let coder = match spawned {
            Ok(handle) => match refs_tx.send(refs) {
                Ok(()) => Coder::Thread { batches: Some(batch_tx), coded: out_rx, handle: Some(handle) },
                Err(mpsc::SendError(refs)) => Coder::Here { refs, done: VecDeque::new(), failed: None },
            },
            // no thread to be had: code the GOP here (the state was never sent)
            Err(_) => Coder::Here { refs, done: VecDeque::new(), failed: None },
        };
        Gop { open: true, sent: 0, taken: 0, coded, dead: false, coder }
    }

    /// Hand a batch to the coder.
    fn send(&mut self, ctx: &Ctx, fixed: FixedQp, batch: Batch) {
        let n = batch.pics.len();
        self.sent += n;
        match &mut self.coder {
            // a coder that died drops the batch; its panic is reported when the GOP is taken
            Coder::Thread { batches, .. } => {
                if let Some(tx) = batches {
                    let _ = tx.send(batch);
                }
            }
            Coder::Here { refs, done, failed } => {
                if failed.is_none() {
                    let recon = batch.recon;
                    match catch_unwind(AssertUnwindSafe(|| encode_batch(ctx, refs, &mut |st, _, _| fixed.frame_qp(st), batch))) {
                        Ok(out) => done.push_back(out.into_iter().map(|c| (c, recon)).collect()),
                        Err(p) => {
                            *failed = Some(panic_text(p));
                            self.dead = true;
                        }
                    }
                }
                self.coded.fetch_add(n, Ordering::Release);
            }
        }
    }

    /// No more batches: the coder finishes once it has coded the ones it has.
    fn close(&mut self) {
        self.open = false;
        if let Coder::Thread { batches, .. } = &mut self.coder {
            *batches = None;
        }
    }

    /// Input pictures sent and not coded yet.
    fn uncoded(&self) -> usize {
        self.sent.saturating_sub(self.coded.load(Ordering::Acquire))
    }

    /// Whether [`Gop::next`] with `wait` returns: something is still to come, or the coder ends.
    fn can_wait(&self) -> bool {
        !self.open || (self.taken < self.sent && !self.dead)
    }

    /// The next coded batch, the end of the GOP, or (without `wait`) nothing yet.
    fn next(&mut self, wait: bool) -> Next {
        match &mut self.coder {
            Coder::Thread { coded, handle, .. } => {
                let got = if wait { coded.recv().map_err(|_| mpsc::TryRecvError::Disconnected) } else { coded.try_recv() };
                match got {
                    Ok(out) => {
                        self.taken += out.len();
                        Next::Coded(out)
                    }
                    Err(mpsc::TryRecvError::Empty) => Next::Pending,
                    // the thread died with the GOP open: the rest of the GOP is lost too
                    Err(mpsc::TryRecvError::Disconnected) if self.open => {
                        self.dead = true;
                        Next::Pending
                    }
                    // the thread ended (closed, cancelled or panicked) and everything it sent was taken
                    Err(mpsc::TryRecvError::Disconnected) => match handle.take() {
                        Some(h) => Next::Finished(h.join().unwrap_or_else(|p| Err(panic_text(p)))),
                        None => Next::Finished(Err("GOP taken twice".into())),
                    },
                }
            }
            Coder::Here { refs, done, failed } => match done.pop_front() {
                Some(out) => {
                    self.taken += out.len();
                    Next::Coded(out)
                }
                None if self.open => Next::Pending,
                None => Next::Finished(match failed.take() {
                    Some(e) => Err(e),
                    None => Ok(std::mem::take(refs)),
                }),
            },
        }
    }
}

/// The GOP thread: code the batches in order as they come, until the GOP is closed.
fn code(ctx: &Ctx, fixed: FixedQp, mut refs: Refs, batches: &mpsc::Receiver<Batch>, out: &mpsc::Sender<Out>, coded: &AtomicUsize, cancel: &AtomicBool) -> Refs {
    for batch in batches {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let (n, recon) = (batch.pics.len(), batch.recon);
        let pics = encode_batch(ctx, &mut refs, &mut |st, _, _| fixed.frame_qp(st), batch);
        coded.fetch_add(n, Ordering::Release);
        if out.send(pics.into_iter().map(|c| (c, recon)).collect()).is_err() {
            break;
        }
    }
    refs
}

/// The message of a panic payload (a `String` or `&str`), or "panic" for any other payload.
fn panic_text(p: Box<dyn Any + Send>) -> String {
    match p.downcast::<String>() {
        Ok(s) => *s,
        Err(p) => p.downcast_ref::<&str>().map_or_else(|| "panic".to_string(), |s| (*s).to_string()),
    }
}

impl Gops {
    /// The GOP-parallel state, or `None` for the serial path: not asked for, a rate control with
    /// feedback, one thread, or two GOPs of `keyint` pictures over the budget.
    pub(crate) fn new(cfg: &EncoderConfig, fixed: Option<FixedQp>, threads: usize, frame_bytes: usize, max_refs: usize) -> Option<Gops> {
        let fixed = fixed?;
        if !cfg.parallel_gops || threads < 2 {
            return None;
        }
        let budget = GOP_BUFFER_BYTES / frame_bytes.max(1);
        let work = 3 * max_refs + usize::from(cfg.bframes) + 4;
        let longest = (cfg.keyint as usize).saturating_add(work);
        // one GOP coding while the next one comes in, at least
        if longest.saturating_mul(2).saturating_add(SPARE_FRAMES) > budget {
            return None;
        }
        Some(Gops {
            fixed,
            gops: VecDeque::new(),
            next_seq: 0,
            idrs: 0,
            refs: 0,
            budget,
            work,
            max_gops: threads,
            analysis: rayon::ThreadPoolBuilder::new()
                .num_threads(threads.div_ceil(6).clamp(2, 4))
                .thread_name(|i| format!("h264enc-analysis-{i}"))
                .build()
                .ok(),
            cancel: Arc::new(AtomicBool::new(false)),
            #[cfg(test)]
            plant_panic: None,
            #[cfg(test)]
            plant_hold: false,
            #[cfg(test)]
            live: Arc::new(AtomicUsize::new(0)),
        })
    }

    /// Padded pictures held with `inputs` more input pictures on the calling thread.
    fn frames_held(&self, inputs: usize) -> usize {
        self.gops.iter().map(|g| g.uncoded() + self.work).sum::<usize>() + inputs + SPARE_FRAMES
    }

    /// The pool that analyses the input pictures, if it could be built.
    pub(crate) fn analysis_pool(&self) -> Option<&rayon::ThreadPool> {
        self.analysis.as_ref()
    }

    /// The open GOP (the newest one, unless a flush closed it).
    fn open(&mut self) -> Option<&mut Gop> {
        self.gops.back_mut().filter(|g| g.open)
    }
}

impl Drop for Gops {
    /// Stop the GOP threads and wait for them: each ends after its current batch at most, so the
    /// GOPs' reference pictures, buffers and pool work are released before the encoder is gone. A
    /// thread that panicked is joined without passing the panic on.
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
        // every channel closes before the first wait: a thread waiting for a batch then ends, and
        // no GOP thread waits for this one
        for g in &mut self.gops {
            g.close();
        }
        for g in &mut self.gops {
            if let Coder::Thread { handle, .. } = &mut g.coder
                && let Some(h) = handle.take()
            {
                // the encoder is going away: the GOP's result, or its panic, has no one to go to
                let _ = h.join();
            }
        }
    }
}

impl Encoder {
    /// Before an input picture: take coded pictures of the oldest GOP (waiting for them) while one
    /// more picture would pass the budget.
    pub(crate) fn gop_make_room(&mut self) {
        while let Some(g) = self.gops.as_ref()
            && g.frames_held(self.pending.len() + 1) > g.budget
        {
            if !self.gop_take_front(true) {
                break;
            }
        }
    }

    /// Hand a batch to its GOP. An IDR batch closes the open GOP and starts the next one, with a
    /// fresh reference state carrying the stream's counters; the first batch after a flush
    /// continues from the state the flush ended with.
    pub(crate) fn gop_schedule(&mut self, batch: Batch) {
        let idr = batch.pics.iter().any(|p| p.idr);
        let Some(g) = self.gops.as_mut() else { return };
        if idr || g.open().is_none() {
            if let Some(open) = g.open() {
                open.close();
            }
            // at most `max_gops` threads: the oldest GOPs are all closed now
            while self.gops.as_ref().is_some_and(|g| g.gops.len() >= g.max_gops) {
                if !self.gop_take_front(true) {
                    break;
                }
            }
            let Some(g) = self.gops.as_mut() else { return };
            let refs = if idr {
                // a state kept from a flush is no longer needed: keep its buffers only
                let stale = self.refs.take_state();
                self.refs.reuse(stale);
                // the identification SEI goes with the first picture of the stream only
                Refs { idr_count: g.idrs, ref_uid: g.refs, sei_sent: batch.first > 0, hpel_pool: std::mem::take(&mut self.refs.hpel_pool), ..Refs::default() }
            } else {
                self.refs.take_state()
            };
            #[cfg(test)]
            let plant = Plant { panic: g.plant_panic == Some(g.next_seq), hold: g.plant_hold, alive: Alive::new(&g.live) };
            g.next_seq += 1;
            let gop = Gop::start(
                &self.ctx,
                g.fixed,
                refs,
                &g.cancel,
                #[cfg(test)]
                plant,
            );
            g.gops.push_back(gop);
        }
        let Some(g) = self.gops.as_mut() else { return };
        for p in &batch.pics {
            g.idrs += u32::from(p.idr);
            // B pictures always find both references in a GOP, so a planned I/P picture is exactly
            // a reference picture (and the numbering only needs to be consistent within a GOP:
            // reference identities are only compared)
            g.refs += u32::from(p.st != SliceType::B);
        }
        let fixed = g.fixed;
        if let Some(open) = g.open() {
            open.send(&self.ctx, fixed, batch);
        }
    }

    /// Take what the oldest GOP has coded (with `wait`, at least one batch or its end); a GOP
    /// that is done leaves the queue, and the stream continues from the last one after a flush.
    fn gop_take_front(&mut self, wait: bool) -> bool {
        let Some(front) = self.gops.as_mut().and_then(|g| g.gops.front_mut()) else { return false };
        if wait && !front.can_wait() {
            return false;
        }
        match front.next(wait) {
            Next::Pending => false,
            Next::Coded(out) => {
                for (c, recon) in out {
                    self.deliver(c, recon);
                }
                true
            }
            Next::Finished(end) => {
                let last = self.gops.as_mut().is_some_and(|g| {
                    g.gops.pop_front();
                    g.gops.is_empty()
                });
                match end {
                    // the GOP a flush ended with: the stream may continue from its references
                    Ok(refs) if last => self.refs.adopt(refs),
                    Ok(refs) => self.refs.reuse(refs),
                    Err(e) => {
                        self.failed.get_or_insert(Error::Worker(e));
                    }
                }
                true
            }
        }
    }

    /// Take what is coded, in decoding order, without waiting.
    pub(crate) fn gop_poll(&mut self) {
        while self.gop_take_front(false) {}
    }

    /// Close the open GOP and take every GOP (flush).
    pub(crate) fn gop_finish(&mut self) {
        if let Some(open) = self.gops.as_mut().and_then(Gops::open) {
            open.close();
        }
        while self.gops.as_ref().is_some_and(|g| !g.gops.is_empty()) {
            if !self.gop_take_front(true) {
                break;
            }
        }
    }

    /// Make the GOP numbered `seq` (0 = the first) panic on its thread.
    #[cfg(test)]
    pub(crate) fn plant_gop_panic(&mut self, seq: u64) {
        if let Some(g) = self.gops.as_mut() {
            g.plant_panic = Some(seq);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::Ordering;

    use crate::{Encoder, EncoderConfig, Error, Packet, YuvFrame};

    /// A slowly moving gradient (no scene cuts).
    fn picture(w: usize, h: usize, t: usize) -> [Vec<u8>; 3] {
        let y = (0..w * h).map(|i| ((i % w + i / w + t) % 256) as u8).collect();
        let c = (0..w * h / 4).map(|i| (128 + (i + t) % 16) as u8).collect::<Vec<u8>>();
        [y, c.clone(), c]
    }

    /// A GOP-parallel encoder (64x48, `keyint` 4, three threads), the GOP numbered `plant` panicking.
    fn encoder(plant: Option<u64>) -> Encoder {
        let mut c = EncoderConfig::new(64, 48, 10, 1);
        c.keyint = 4;
        c.threads = 3;
        let mut enc = Encoder::new(c).unwrap();
        assert!(enc.gop_parallel());
        if let Some(seq) = plant {
            enc.plant_gop_panic(seq);
        }
        enc
    }

    /// Encode 20 pictures; the packets and the number of failed calls.
    fn run(mut enc: Encoder) -> (Vec<Packet>, usize) {
        let (mut packets, mut failures) = (Vec::new(), 0);
        let mut take = |r: Result<Vec<Packet>, Error>, packets: &mut Vec<Packet>| match r {
            Ok(p) => {
                packets.extend(p);
                true
            }
            Err(Error::Worker(msg)) => {
                assert!(msg.contains("planted GOP panic"), "{msg}");
                failures += 1;
                false
            }
            Err(e) => panic!("{e}"),
        };
        for t in 0..20 {
            let [y, u, v] = picture(64, 48, t);
            let frame = YuvFrame { y: &y, u: &u, v: &v, y_stride: 64, uv_stride: 32 };
            take(enc.try_encode(&frame, t as i64), &mut packets);
        }
        // a failure reported by the flush keeps the other packets for the next call
        while !take(enc.flush(), &mut packets) {}
        (packets, failures)
    }

    /// A panic on one GOP thread is reported once as `Error::Worker`; the packets of that GOP are
    /// lost and every other packet comes out unchanged.
    #[test]
    fn a_panicking_gop_is_an_error_and_the_other_gops_still_come() {
        let (all, failures) = run(encoder(None));
        assert_eq!(failures, 0);
        let (rest, failures) = run(encoder(Some(1)));
        assert_eq!(failures, 1, "one error, once");
        // GOP 1 (pictures 4..8) is lost; every other packet is there, unchanged
        let expected: Vec<&Packet> = all.iter().filter(|p| !(4..8).contains(&p.pts)).collect();
        assert_eq!(rest.len(), expected.len());
        assert_eq!(all.len(), expected.len() + 4);
        for (a, b) in rest.iter().zip(expected) {
            assert!(a.data == b.data && (a.pts, a.dts) == (b.pts, b.dts));
        }
    }

    /// Dropping an encoder whose GOPs still have batches to code returns.
    #[test]
    fn dropping_an_encoder_with_gops_running_returns() {
        let mut enc = encoder(None);
        for t in 0..40 {
            let [y, u, v] = picture(64, 48, t);
            let frame = YuvFrame { y: &y, u: &u, v: &v, y_stride: 64, uv_stride: 32 };
            enc.try_encode(&frame, t as i64).unwrap();
        }
        drop(enc);
    }

    /// Dropping the encoder waits for its GOP threads: every one was still busy with a batch when
    /// the drop began, and none is left once it returns.
    #[test]
    fn no_gop_thread_outlives_its_encoder() {
        let mut enc = encoder(None);
        let live = {
            let g = enc.gops.as_mut().unwrap();
            g.plant_hold = true;
            Arc::clone(&g.live)
        };
        for t in 0..12 {
            let [y, u, v] = picture(64, 48, t);
            let frame = YuvFrame { y: &y, u: &u, v: &v, y_stride: 64, uv_stride: 32 };
            enc.try_encode(&frame, t as i64).unwrap();
        }
        let (gops, busy) = {
            let gops = &enc.gops.as_ref().unwrap().gops;
            (gops.len(), gops.iter().all(|g| g.uncoded() > 0))
        };
        assert!(gops >= 2 && busy, "{gops} GOPs, all still coding: {busy}");
        assert_eq!(live.load(Ordering::SeqCst), gops);
        drop(enc);
        assert_eq!(live.load(Ordering::SeqCst), 0, "GOP threads left after the drop");
    }
}
