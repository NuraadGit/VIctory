//! Command queues. Submitted lists run on the host GPU and are waited for,
//! so fence signals complete at once.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, QUEUE, S_OK, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

pub(crate) const QUEUE_IIDS: [Iid; 4] = [
    guid(
        0x0ec870a6,
        0x5d7e,
        0x4c22,
        [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct QueueDesc {
    kind: u32,
    priority: i32,
    flags: u32,
    nodes: u32,
}

/// The title reads the list type right after the header.
#[repr(C)]
struct Queue {
    object: Object,
    kind: u32,
    desc: QueueDesc,
}
const _: () = assert!(std::mem::offset_of!(Queue, kind) == 0x10);

fn queue<'a>(this: This) -> &'a Queue {
    // SAFETY: called from queue methods only.
    unsafe { &*this.cast::<Queue>() }
}

pub(crate) extern "system" fn create(
    _: This,
    desc: *const QueueDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(?desc, "command queue");
    let queue = Box::new(Queue {
        object: if crate::device::newer() {
            Object::new(&NEWER_FUNCTIONS)
        } else {
            Object::new(&FUNCTIONS)
        },
        kind: desc.kind,
        desc,
    });
    object::hand_out(QUEUE, queue, iid, &QUEUE_IIDS, out)
}

/// The console's queue description: an engine or pipe and a queue on it
/// in place of PC's priority and node mask.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct QueueDescX {
    kind: u32,
    flags: u32,
    engine_or_pipe: u32,
    queue: u32,
}

/// Device method `CreateCommandQueueX`. The host picks engines itself, so
/// the queue is created as on PC.
pub(crate) extern "system" fn create_x(
    this: This,
    desc: *const QueueDescX,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&x) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(desc = ?x, "console command queue");
    let desc = QueueDesc {
        kind: x.kind,
        priority: 0,
        flags: x.flags,
        nodes: 1,
    };
    create(this, &desc, iid, out)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        QUEUE,
        iid,
        object::is(iid, &QUEUE_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Queue>(this)
}

/// Runs the lists' host lists and waits for them (decision 0007), unless
/// one waits for a memory value no submission has written yet; that list
/// and what follows it on this queue are held back ([`held`]).
pub(crate) extern "system" fn execute(this: This, count: u32, lists: *const This) {
    if lists.is_null() {
        return;
    }
    #[cfg(windows)]
    if crate::host::host().is_some() {
        // SAFETY: the title passes `count` lists.
        let lists = unsafe { std::slice::from_raw_parts(lists, count as usize) };
        held::submit(this, held::Held::Lists(lists.iter().map(|&l| l as usize).collect()));
        return;
    }
    let _ = this;
    static ONCE: AtomicBool = AtomicBool::new(false);
    if !ONCE.swap(true, Ordering::Relaxed) {
        tracing::warn!(count, "command lists submitted; no host GPU runs them");
    }
}

/// A list's GPU wait for a memory value (`Wait32BitValueX`,
/// `Wait64BitValueX`): over once `(value & mask) <comparison> reference`,
/// the comparison a `D3D12_COMPARISON_FUNC` (as AMD's `WAIT_REG_MEM`).
#[cfg(windows)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Wait {
    pub(crate) address: u64,
    /// 4 or 8.
    pub(crate) bytes: usize,
    pub(crate) comparison: u32,
    pub(crate) reference: u64,
    pub(crate) mask: u64,
}

#[cfg(windows)]
impl Wait {
    /// Whether the wait is over now. Memory that is not committed and
    /// comparisons we do not know end it, rather than hold a queue forever.
    pub(crate) fn met(&self) -> bool {
        if !crate::texture::committed(self.address as *const u8, self.bytes) {
            return true;
        }
        // SAFETY: committed title memory, `bytes` long.
        let value = unsafe {
            if self.bytes == 4 {
                u64::from((self.address as *const u32).read_volatile())
            } else {
                (self.address as *const u64).read_volatile()
            }
        };
        let (v, r) = (value & self.mask, self.reference);
        match self.comparison {
            1 => false,
            2 => v < r,
            3 => v == r,
            4 => v <= r,
            5 => v > r,
            6 => v != r,
            7 => v >= r,
            _ => true,
        }
    }
}

/// Work held back on queues behind a list that waits for a memory value
/// (#700): lists run one submission after another, and a value another
/// queue or the CPU writes later cannot be waited for inside one. So a
/// submission runs up to its first list whose wait is not over yet; that
/// list, the rest of the submission and every later submission and signal
/// on the same queue wait in order, without blocking the title's thread.
/// They run as soon as the wait is over: at any submission or signal, and
/// every millisecond while anything is held (for values the CPU writes).
///
/// shortcut: a list waits as a whole, so its commands before the wait wait
/// too (they would deadlock only if what the wait needs depended on them).
/// Presents and page mappings on a queue holding work do not wait for it.
#[cfg(windows)]
pub(crate) mod held {
    use std::collections::VecDeque;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

    use crate::object::{self, This};

    /// A wait not over after this long is taken as over: a title would
    /// hang on the console too, but here it more likely means a value we do
    /// not model (a write on a path we do not see).
    const GIVE_UP: Duration = Duration::from_secs(2);

    /// A queue's work, objects as addresses (held ones with a reference).
    pub(crate) enum Held {
        Lists(Vec<usize>),
        Signal(usize, u64),
    }

    struct Queue {
        queue: usize,
        /// Since the first held item last moved.
        since: Instant,
        held: VecDeque<Held>,
    }

    static HELD: Mutex<Vec<Queue>> = Mutex::new(Vec::new());
    /// The thread that retries held work while there is any.
    static RETRYING: AtomicBool = AtomicBool::new(false);

    /// Runs `work` for `queue`, or holds it back behind what the queue
    /// already holds or behind a wait in it.
    pub(crate) fn submit(queue: This, work: Held) {
        let mut held = HELD.lock().unwrap_or_else(|e| e.into_inner());
        let queue = queue as usize;
        match held.iter_mut().find(|q| q.queue == queue) {
            Some(q) => {
                keep(&work);
                q.held.push_back(work);
            }
            None => {
                let rest = run(work, false);
                if let Some(rest) = rest {
                    keep(&rest);
                    held.push(Queue {
                        queue,
                        since: Instant::now(),
                        held: VecDeque::from([rest]),
                    });
                }
            }
        }
        retry(&mut held);
        if !held.is_empty() && !RETRYING.swap(true, Ordering::AcqRel) {
            std::thread::spawn(retry_while_held);
        }
    }

    /// The title resets `list`, which D3D12 allows as soon as it is
    /// submitted: held work naming it runs first, up to and including it and
    /// whole, so the host executes the commands the title submitted and not
    /// the ones it records next. Executing a list the title has reopened is
    /// an invalid call, and the device is removed.
    pub(crate) fn list_reset(list: This) {
        let list = list as usize;
        let mut held = HELD.lock().unwrap_or_else(|e| e.into_inner());
        for q in held.iter_mut() {
            let Some(at) = q
                .held
                .iter()
                .position(|w| matches!(w, Held::Lists(l) if l.contains(&list)))
            else {
                continue;
            };
            let due: Vec<Held> = q.held.drain(..=at).collect();
            for work in due {
                let _ = run_kept(work, true);
            }
            q.since = Instant::now();
        }
        held.retain(|q| !q.held.is_empty());
    }

    /// Whether `queue` holds work (for tests).
    #[cfg(test)]
    pub(crate) fn holds(queue: This) -> bool {
        let held = HELD.lock().unwrap_or_else(|e| e.into_inner());
        held.iter().any(|q| q.queue == queue as usize)
    }

    fn retry_while_held() {
        loop {
            std::thread::sleep(Duration::from_millis(1));
            let mut held = HELD.lock().unwrap_or_else(|e| e.into_inner());
            retry(&mut held);
            if held.is_empty() {
                RETRYING.store(false, Ordering::Release);
                return;
            }
        }
    }

    /// Runs what has become ready on every queue, until nothing moves (one
    /// queue's lists may end another's wait).
    fn retry(held: &mut Vec<Queue>) {
        let mut moved = true;
        while moved {
            moved = false;
            for q in held.iter_mut() {
                while let Some(work) = q.held.pop_front() {
                    let late = q.since.elapsed() > GIVE_UP;
                    let before = count(&work);
                    let rest = run_kept(work, late);
                    let after = rest.as_ref().map_or(0, count);
                    if let Some(rest) = rest {
                        q.held.push_front(rest);
                    }
                    if after == before {
                        break;
                    }
                    moved = true;
                    q.since = Instant::now();
                }
            }
            held.retain(|q| !q.held.is_empty());
        }
    }

    fn count(work: &Held) -> usize {
        match work {
            Held::Lists(lists) => lists.len(),
            Held::Signal(..) => 1,
        }
    }

    /// Takes a reference to what `work` names while it is held.
    fn keep(work: &Held) {
        match work {
            Held::Lists(lists) => lists.iter().for_each(|&l| {
                object::add_ref(l as This);
            }),
            Held::Signal(fence, _) => {
                object::add_ref(*fence as This);
            }
        }
    }

    fn drop_kept(objects: &[usize]) {
        for &o in objects {
            object::release_ref(o as This);
        }
    }

    /// [`run`] for held work: the references taken for what ran go.
    fn run_kept(work: Held, late: bool) -> Option<Held> {
        let all: Vec<usize> = match &work {
            Held::Lists(lists) => lists.clone(),
            Held::Signal(fence, _) => vec![*fence],
        };
        let rest = run(work, late);
        let left = rest.as_ref().map_or(0, count);
        drop_kept(&all[..all.len() - left]);
        rest
    }

    /// Runs `work` up to its first list whose wait is not over (all of it
    /// when `late`); what is left.
    fn run(work: Held, late: bool) -> Option<Held> {
        match work {
            Held::Signal(fence, value) => {
                crate::fence::signal(fence as This, value);
                None
            }
            Held::Lists(lists) => {
                let ready = if late {
                    lists.len()
                } else {
                    lists
                        .iter()
                        .position(|&l| {
                            crate::list::host_list(l as This).is_some_and(|h| !h.waits_over())
                        })
                        .unwrap_or(lists.len())
                };
                if late {
                    tracing::warn!(
                        lists = lists.len(),
                        "a list's memory wait not over after 2 s; running it anyway"
                    );
                }
                if ready > 0
                    && let Some(host) = crate::host::host()
                {
                    let runs: Vec<_> = lists[..ready]
                        .iter()
                        .filter_map(|&l| crate::list::host_list(l as This))
                        .collect();
                    host.execute(&runs);
                }
                (ready < lists.len()).then(|| Held::Lists(lists[ready..].to_vec()))
            }
        }
    }
}

/// The start of a present plane: frame token, then the plane's resources.
#[repr(C)]
pub(crate) struct Plane {
    token: u64,
    count: u32,
    resources: *const This,
}
/// Submission tagged with the frame it belongs to `(count, lists, frame
/// token, flags)`; frames are paced elsewhere, so it runs as a plain one.
extern "system" fn execute_x(this: This, count: u32, lists: *const This, _token: u64, _flags: u32) {
    execute(this, count, lists);
}
/// Signal with sync settings for the top and bottom of the pipe; our
/// submissions are complete when they return, so it signals at once.
extern "system" fn signal2_x(
    this: This,
    fence: This,
    value: u64,
    _bottom: *const u8,
    _top: *const u8,
) -> HRESULT {
    signal(this, fence, value)
}
/// Markers and events only matter to profilers.
extern "system" fn marker(_: This) {}
/// Signals at once, or after the work the queue holds back ([`held`]).
pub(crate) extern "system" fn signal(this: This, fence: This, value: u64) -> HRESULT {
    if fence.is_null() {
        return E_INVALIDARG;
    }
    #[cfg(windows)]
    if crate::host::host().is_some() {
        held::submit(this, held::Held::Signal(fence as usize, value));
        return S_OK;
    }
    let _ = this;
    crate::fence::signal(fence, value)
}
/// Nothing runs on the queue, so waiting for a fence never blocks it.
extern "system" fn wait(_: This, fence: This, _value: u64) -> HRESULT {
    if fence.is_null() { E_INVALIDARG } else { S_OK }
}
/// Ticks per second of GPU timestamps (see [`Written::GpuTimestamp`]).
extern "system" fn timestamp_frequency(_: This, out: *mut u64) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null.
    unsafe { out.write(TIMESTAMP_FREQUENCY) };
    S_OK
}
const TIMESTAMP_FREQUENCY: u64 = 100_000_000;

/// What a list's bottom-of-pipe write stores when the list has run.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Written {
    Value(u64),
    /// The GPU clock: [`TIMESTAMP_FREQUENCY`] ticks per second since the
    /// process started (our GPU work is done on the CPU's timeline).
    GpuTimestamp,
    /// The CPU's performance counter.
    SystemTimestamp,
}

impl Written {
    /// Stores the low `bytes` (4 or 8) of the value at `at`.
    pub(crate) fn write(self, at: *mut u8, bytes: usize) {
        let value = match self {
            Written::Value(v) => v,
            Written::GpuTimestamp => {
                static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
                let elapsed = START.get_or_init(std::time::Instant::now).elapsed();
                (elapsed.as_nanos() / (1_000_000_000 / u128::from(TIMESTAMP_FREQUENCY))) as u64
            }
            Written::SystemTimestamp => system_timestamp(),
        };
        // SAFETY: the title names writable GPU memory (unified with ours).
        unsafe {
            if bytes == 4 {
                at.cast::<u32>().write_unaligned(value as u32);
            } else {
                at.cast::<u64>().write_unaligned(value);
            }
        }
    }
}

fn system_timestamp() -> u64 {
    #[cfg(windows)]
    {
        let mut count = 0;
        // SAFETY: a valid out pointer.
        unsafe { windows_sys::Win32::System::Performance::QueryPerformanceCounter(&mut count) };
        count as u64
    }
    #[cfg(not(windows))]
    0
}

/// Frames presented so far.
static FRAMES: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "diagnostics")]
pub(crate) fn presented() -> u64 {
    FRAMES.load(Ordering::Relaxed)
}

/// Shows the first plane's first resource in our window.
///
/// shortcut: one plane, no scaling or placement, the source expected in
/// the common state.
extern "system" fn present(
    _: This,
    planes: u32,
    plane_parameters: *const Plane,
    _parameters: *const u8,
) -> HRESULT {
    let frame = FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
    if frame.is_power_of_two() {
        let (draws, dispatches) = crate::list::take_counts();
        #[cfg(windows)]
        let (drawn, dispatched) = {
            static DRAWN: AtomicU64 = AtomicU64::new(0);
            static DISPATCHED: AtomicU64 = AtomicU64::new(0);
            let since = |last: &AtomicU64, now: u64| now - last.swap(now, Ordering::Relaxed);
            (
                since(&DRAWN, crate::draw::drawn()),
                since(&DISPATCHED, crate::draw::dispatched()),
            )
        };
        #[cfg(not(windows))]
        let (drawn, dispatched) = (0, 0);
        tracing::info!(
            frame,
            draws,
            drawn,
            dispatches,
            dispatched,
            "frame presented (since the last line: draws and dispatches, and those run on the host)"
        );
        #[cfg(windows)]
        if frame >= 256 {
            let (hits, misses) = crate::shader_cache::counts();
            tracing::info!(hits, misses, "shaders from the cache and translated so far");
        }
    }
    #[cfg(all(windows, feature = "diagnostics"))]
    if crate::capture::listed(frame) {
        crate::capture::dump_targets(frame);
    }
    #[cfg(windows)]
    if planes > 0
        // SAFETY: the title passes `planes` plane descriptions.
        && let Some(plane) = unsafe { plane_parameters.as_ref() }
        && plane.count > 0
        && !plane.resources.is_null()
        // SAFETY: the plane lists `count` resources.
        && let resource = unsafe { *plane.resources }
        && let Some(texture) = crate::resource::host_resource(resource)
        && let Some(host) = crate::host::host()
    {
        const COPY_SOURCE: u32 = 0x800;
        let state = crate::resource::host_transition(resource, COPY_SOURCE)
            .map_or(COPY_SOURCE, |(_, before)| before);
        let shown = host.present(texture, state, frame);
        crate::resource::host_transition(resource, state);
        #[cfg(feature = "diagnostics")]
        blinks(frame, resource, state, shown);
        #[cfg(not(feature = "diagnostics"))]
        let _ = shown;
    }
    #[cfg(not(windows))]
    let _ = (planes, plane_parameters);
    S_OK
}
/// Diagnostics: logs a presented frame that is black after one that was
/// not, with what was presented and how many draws ran since.
#[cfg(all(windows, feature = "diagnostics"))]
fn blinks(frame: u64, resource: This, state: u32, shown: Option<bool>) {
    static LAST: std::sync::Mutex<(bool, u64, u64)> = std::sync::Mutex::new((false, 0, 0));
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let drawn = crate::draw::drawn();
    let black = shown.unwrap_or(false);
    let draws = crate::draw::take_frame();
    if black && !last.0 {
        last.2 += 1;
        tracing::warn!(
            frame,
            resource = ?resource,
            state = format_args!("{state:#x}"),
            draws = drawn - last.1,
            copied = shown.is_some(),
            black_frames = last.2,
            levels = ?crate::capture::recent_levels(),
            "a black frame after a frame that was not"
        );
        for d in draws {
            tracing::warn!("black frame {frame}: {d}");
        }
        AFTER.store(frame + 6, Ordering::Relaxed);
    }
    // Six frames on: the levels around it (a fade or a single drop).
    static AFTER: AtomicU64 = AtomicU64::new(0);
    if AFTER.load(Ordering::Relaxed) == frame {
        tracing::warn!(frame, levels = ?crate::capture::recent_levels(), "levels after the black frame");
    }
    *last = (black, drawn, last.2);
}

/// A run of destination pages: how they are mapped (0 pool pages from
/// `start` on, 1 pool page `start` each, 2 unmapped, 3 left as they are)
/// and how many.
#[repr(C)]
pub(crate) struct PageRange {
    kind: u32,
    start: u32,
    count: u32,
}

/// The ranges mapped from a destination address on.
#[repr(C)]
pub(crate) struct PageBatch {
    destination: u64,
    count: u32,
    ranges: *const PageRange,
}

/// Bytes per mapped page.
const PAGE: u64 = 64 << 10;

/// `CopyPageMappingsBatchX(count, batches, pool, pool pages, flags)`: maps
/// pages of a pool (memory the title mapped physical pages into) at the
/// batches' addresses. Grand Theft Auto VI streams textures into pool pages
/// and maps them into its tiled heap this way (title+0xf2c470, ranges of
/// one constant page or of unmapped pages). Done at once: every earlier
/// submission has run.
extern "system" fn copy_page_mappings(
    _: This,
    count: u32,
    batches: *const PageBatch,
    pool: u64,
    pool_pages: u32,
    _flags: u32,
) -> HRESULT {
    let Some(map) = crate::resource::page_mapper() else {
        return E_INVALIDARG;
    };
    if batches.is_null() && count > 0 {
        return E_INVALIDARG;
    }
    let batches = (count > 0).then(|| {
        // SAFETY: the title passes `count` batches.
        unsafe { std::slice::from_raw_parts(batches, count as usize) }
    });
    let mut ok = true;
    for b in batches.into_iter().flatten() {
        if b.ranges.is_null() {
            continue;
        }
        // SAFETY: each batch holds `count` ranges.
        let ranges = unsafe { std::slice::from_raw_parts(b.ranges, b.count as usize) };
        let mut at = b.destination;
        for r in ranges {
            let len = u64::from(r.count) * PAGE;
            let inside = |pages: u32| r.start.checked_add(pages).is_some_and(|e| e <= pool_pages);
            let source = pool + u64::from(r.start) * PAGE;
            ok &= match r.kind {
                0 if inside(r.count) => map(at, source, len),
                1 if inside(1) => (0..u64::from(r.count)).all(|k| map(at + k * PAGE, source, PAGE)),
                2 => map(at, 0, len),
                3 => true,
                _ => false,
            };
            at += len;
        }
    }
    if !ok {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!(pool = format_args!("{pool:#x}"), "a page mapping failed"));
    }
    S_OK
}

extern "system" fn desc(this: This, out: *mut QueueDesc) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(queue(this).desc) };
    }
}

/// The queue methods by their slot in older titles' queues (the 2020-era
/// layout): IUnknown 3, object 4, device child 1, queue 26 (12 as on PC,
/// then console ones).
const METHODS: [(usize, *const ()); 15] = [
    (0, query as *const ()),
    (2, release as *const ()),
    (7, object::parent_device as *const ()), // 0x38
    (10, execute as *const ()),              // 0x50
    (11, marker as *const ()),               // 0x58 set marker
    (12, marker as *const ()),               // 0x60 begin event
    (13, marker as *const ()),               // 0x68 end event
    (14, signal as *const ()),               // 0x70
    (15, wait as *const ()),                 // 0x78
    (16, timestamp_frequency as *const ()),  // 0x80
    (26, desc as *const ()),                 // 0xd0
    (29, copy_page_mappings as *const ()),   // 0xe8 page mapping batch
    (30, present as *const ()),              // 0xf0
    (33, execute_x as *const ()),            // 0x108
    (32, signal2_x as *const ()),            // 0x100
];

/// Newer titles' queues have two more methods before Signal, which moved
/// from 0x70 to 0x80: every slot from 0x70 on is two later (taken here as
/// starting exactly at 0x70, the methods before it kept in place).
/// Inferred from Grand Theft Auto VI's calls: Signal 0x80 `(fence,
/// value)`, timestamp frequency 0x90 (title+0x24841a6), suspend and resume
/// 0xb8/0xc0, PresentX 0x100, the frame-token submission 0x118 `(count,
/// lists, token, flags)` (title+0x24821dd).
const fn newer_slot(slot: usize) -> usize {
    if slot >= 14 { slot + 2 } else { slot }
}

const fn queue_table(newer: bool) -> [*const (); 36] {
    let mut f = object::d3d12_object(object::missing_table::<QUEUE, 36>());
    let mut i = 0;
    while i < METHODS.len() {
        let (slot, method) = METHODS[i];
        f[if newer { newer_slot(slot) } else { slot }] = method;
        i += 1;
    }
    f
}

static FUNCTIONS: Table<36> = Table(queue_table(false));
static NEWER_FUNCTIONS: Table<36> = Table(queue_table(true));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;

    /// Where a queue's signal is: newer layouts have it two slots later.
    /// Another test may switch layouts while this one runs, so the queue's
    /// own table decides.
    fn signal_slot(q: This) -> usize {
        // SAFETY: one of our queues; its first field is the table.
        let table = unsafe { *q.cast::<*const *const ()>() };
        if table == NEWER_FUNCTIONS.0.as_ptr() {
            0x80
        } else {
            0x70
        }
    }

    #[test]
    fn queue_signals_complete_at_once() {
        let null = std::ptr::null();
        let desc = QueueDesc {
            kind: 0,
            priority: 0,
            flags: 0,
            nodes: 0,
        };
        let mut q = null;
        assert_eq!(create(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        // SAFETY: the title reads the type at +0x10.
        assert_eq!(unsafe { q.cast::<u8>().add(0x10).cast::<u32>().read() }, 0);
        let mut fence = null;
        let fence_iid = &crate::fence::FENCE_IIDS[0];
        assert_eq!(
            crate::fence::create(null, 0, 0, fence_iid, &mut fence),
            S_OK
        );
        let signal = slot::<extern "system" fn(This, This, u64) -> HRESULT>(q, signal_slot(q));
        assert_eq!(signal(q, fence, 7), S_OK);
        assert_eq!(
            slot::<extern "system" fn(This) -> u64>(fence, 0x40)(fence),
            7
        );
        assert_eq!(slot::<extern "system" fn(This) -> u32>(q, 0x10)(q), 0);
    }

    /// Grand Theft Auto VI's compute queue: `{COMPUTE, 0, pipe 1, queue 0}`.
    #[test]
    fn console_queues_are_created_as_on_pc() {
        let null = std::ptr::null();
        let desc = QueueDescX {
            kind: 2,
            flags: 0,
            engine_or_pipe: 1,
            queue: 0,
        };
        let mut q = null;
        assert_eq!(create_x(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        assert_eq!(queue(q).desc.kind, 2);
        let none = std::ptr::null();
        assert_eq!(create_x(null, none, &QUEUE_IIDS[0], &mut q), E_INVALIDARG);
    }

    #[test]
    fn newer_queues_have_their_methods_two_slots_later() {
        let (old, new) = (queue_table(false), queue_table(true));
        assert_eq!(old[10], new[10]); // submission stays at 0x50
        assert_eq!(old[14], new[16]); // signal 0x70 → 0x80
        assert_eq!(old[30], new[32]); // present 0xf0 → 0x100
        assert_eq!(new[35], execute_x as *const ()); // 0x118
    }

    #[test]
    fn timestamps_and_values_are_written() {
        let mut at = [0u64; 2];
        Written::Value(0x1_0000_0002).write(at.as_mut_ptr().cast(), 4);
        assert_eq!(at[0], 2);
        Written::GpuTimestamp.write(at[1..].as_mut_ptr().cast(), 8);
        let first = at[1];
        std::thread::sleep(std::time::Duration::from_millis(2));
        Written::GpuTimestamp.write(at[1..].as_mut_ptr().cast(), 8);
        assert!(at[1] > first);
    }

    #[test]
    #[cfg(windows)]
    fn memory_waits_compare_as_d3d12_does() {
        let word = Box::new(0x1_0000_0005u64);
        let wait = |bytes, comparison, reference, mask| Wait {
            address: (&raw const *word) as u64,
            bytes,
            comparison,
            reference,
            mask,
        };
        // The whole word against 0x1_0000_0005.
        let full = u64::MAX;
        assert!(wait(8, 7, 0x1_0000_0005, full).met(), "greater or equal");
        assert!(!wait(8, 7, 0x1_0000_0006, full).met());
        assert!(wait(8, 3, 0x1_0000_0005, full).met(), "equal");
        assert!(wait(8, 2, 0x1_0000_0006, full).met(), "less");
        assert!(!wait(8, 5, 0x1_0000_0005, full).met(), "greater");
        assert!(!wait(8, 1, 0, full).met(), "never");
        assert!(wait(8, 8, 0, full).met(), "always");
        assert!(wait(8, 0, 0, full).met(), "unknown comparisons end the wait");
        // The low dword, masked.
        assert!(wait(4, 3, 5, full).met());
        assert!(wait(4, 3, 4, 4).met());
        // Memory that is not there ends the wait.
        let mut gone = wait(8, 1, 0, full);
        gone.address = 0x10;
        assert!(gone.met());
    }
}
