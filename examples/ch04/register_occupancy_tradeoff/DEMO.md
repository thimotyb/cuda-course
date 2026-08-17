# Demo: Register Pressure, Occupancy, and Profiler Evidence

This demo shows how increasing per-thread register use can reduce occupancy. It uses the same kernel with the same problem size and block size, then compares two register-pressure levels with Nsight Compute CLI.

Important caveat: in this teaching kernel, a higher register-pressure level also performs more arithmetic per thread because more private values are updated. Use the profiler data to demonstrate the resource/occupancy trade-off. Use timing as the observed outcome for this specific kernel, not as proof that occupancy alone caused the full runtime difference.

## 1. Build

```bash
cd examples/ch04/register_occupancy_tradeoff
nvcc register_occupancy_tradeoff.cu -O3 -lineinfo -o register_occupancy_tradeoff
```

## 2. Baseline Timing Without Profiler

Run the low-register-pressure case:

```bash
./register_occupancy_tradeoff 16777216 256 32 128 20 0
```

Run the high-register-pressure case:

```bash
./register_occupancy_tradeoff 16777216 256 128 128 20 0
```

What to record from the program output:

```text
actual regs
blocks/SM
warps/SM
occ %
avg ms
Gelem/s
```

Example values measured on the local RTX 5060 Ti:

| Case | Command level | Actual regs/thread | Blocks/SM | Warps/SM | Theoretical occupancy | Avg kernel time | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Low register pressure | `32` | `40` | `6` | `48` | `100.0%` | `12.4215 ms` | `1.35 Gelem/s` |
| High register pressure | `128` | `134` | `1` | `8` | `16.7%` | `30.4934 ms` | `0.55 Gelem/s` |

How to comment:

The two runs use the same `N` and the same `256` threads per block. The low-pressure kernel uses `40` registers per thread, so the SM can keep `6` blocks resident, or `48` warps. The high-pressure kernel uses `134` registers per thread, so the register file can hold only `1` block, or `8` warps. This is the core trade-off: registers are fast, but they are also an occupancy-limiting resource.

Do not say: higher occupancy always means faster.

Say instead: in this run, higher register pressure reduces occupancy substantially and the measured kernel is slower. The profiler confirms that the limiting launch resource changed to registers. For a different kernel, more registers could still be beneficial if they avoid spilling or reduce memory traffic.

## 3. Profile With Nsight Compute CLI

Profile the low-register-pressure case:

```bash
ncu \
  --section LaunchStats \
  --section Occupancy \
  --section SchedulerStats \
  --section WarpStateStats \
  --target-processes all \
  --force-overwrite \
  --export reg-occ-low \
  ./register_occupancy_tradeoff 16777216 256 32 128 5 0
```

Profile the high-register-pressure case:

```bash
ncu \
  --section LaunchStats \
  --section Occupancy \
  --section SchedulerStats \
  --section WarpStateStats \
  --target-processes all \
  --force-overwrite \
  --export reg-occ-high \
  ./register_occupancy_tradeoff 16777216 256 128 128 5 0
```

These commands create:

```text
reg-occ-low.ncu-rep
reg-occ-high.ncu-rep
```

Do not use the runtime printed while running under `ncu` as a normal performance number. Nsight Compute replays kernels over multiple passes, so profiler runtime is intentionally distorted. Use non-profiled runs for timing.

## 4. Extract Key Metrics From Reports

Use this helper command to extract the metrics used in class:

```bash
python3 - <<'PY'
import csv
import subprocess

reports = [
    ("low", "reg-occ-low.ncu-rep"),
    ("high", "reg-occ-high.ncu-rep"),
]

metrics = [
    "Kernel Name",
    "Block Size",
    "Grid Size",
    "launch__registers_per_thread",
    "launch__registers_per_thread_allocated",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_warps",
    "launch__occupancy_limit_blocks",
    "sm__maximum_warps_avg_per_active_cycle",
    "sm__maximum_warps_per_active_cycle_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.per_cycle_active",
    "smsp__warps_active.avg.per_cycle_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
]

for label, report in reports:
    out = subprocess.check_output(
        ["ncu", "--import", report, "--page", "raw", "--csv"],
        text=True,
    )
    lines = out.splitlines()
    header = next(csv.reader([lines[0]]))
    values = next(csv.reader([lines[2]]))
    row = dict(zip(header, values))

    print(f"[{label}] {report}")
    for metric in metrics:
        print(f"{metric}: {row.get(metric, '')}")
    print()
PY
```

## 5. Profiler Values Measured Locally

The following values were measured on the local `NVIDIA GeForce RTX 5060 Ti`.

| Metric | Low pressure, level `32` | High pressure, level `128` | How to explain it |
| --- | ---: | ---: | --- |
| Kernel variant | `registerPressureKernel<32>` | `registerPressureKernel<128>` | Same source template, different compile-time register-pressure level. |
| Block size | `256` | `256` | Same launch shape, so the comparison is focused on per-thread resource use. |
| Grid size | `65536` | `65536` | Same number of blocks. |
| Registers per thread | `40` | `134` | The compiler reports much higher register use in the high-pressure variant. |
| Allocated registers per thread | `40` | `136` | Allocation can be rounded by the hardware/compiler. |
| Occupancy limit from registers | `6 blocks` | `1 block` | This is the key row: registers reduce resident blocks per SM. |
| Occupancy limit from warps | `6 blocks` | `6 blocks` | Warps are not the new bottleneck; registers are. |
| Occupancy limit from max blocks | `24 blocks` | `24 blocks` | The architectural block limit is not binding. |
| Max warps per active cycle | `48` | `8` | The high-register case can keep far fewer warps resident. |
| Max warp occupancy | `100.0%` | `16.7%` | The theoretical occupancy ceiling collapses because of register use. |
| Achieved active warps | `95.4%` | `16.6%` | The measured active-warp level follows the theoretical limit closely. |
| SM active warps/cycle | `45.80` | `7.98` | Same story in absolute warp count. |
| SMSP active warps/cycle | `11.44` | `2.00` | Per scheduler partition, active warp supply is much lower. |
| Eligible warps/cycle | `9.21` | `1.82` | The scheduler has fewer ready warps to choose from. |
| Issue active | `99.8%` | `93.7%` | Both issue frequently, but the high-pressure case has a much smaller pool of warps. |
| Long scoreboard stall ratio | `0.11` | `0.05` | This kernel is mostly arithmetic-heavy, so memory latency is not the main story here. |
| Math pipe throttle ratio | `1.95` | `0.00` | Low-pressure has many active warps competing for math pipeline issue slots. |
| Not selected stall ratio | `8.24` | `0.94` | Low-pressure has many ready warps; many are ready but not selected in a given cycle. |
| Wait stall ratio | `0.10` | `0.08` | Not the primary teaching signal in this run. |

## 6. How to Present the Trade-off

Start from the resource equation:

```text
registers_per_block = registers_per_thread x threads_per_block
```

Then apply it to the two runs:

```text
low:  40 regs/thread  x 256 threads/block = 10,240 regs/block
high: 134 regs/thread x 256 threads/block = 34,304 regs/block
```

On this GPU, the register file budget allows the low-pressure variant to keep `6` blocks resident per SM, but the high-pressure variant only `1` block resident per SM.

Then connect blocks to warps:

```text
256 threads/block = 8 warps/block

low:  6 blocks/SM x 8 warps/block = 48 warps/SM
high: 1 block/SM  x 8 warps/block = 8 warps/SM
```

That gives:

```text
low:  48 / 48 = 100.0% theoretical occupancy
high:  8 / 48 = 16.7% theoretical occupancy
```

The profiler confirms the same picture:

```text
low achieved active warps:  ~95.4%
high achieved active warps: ~16.6%
```

Suggested spoken comment:

The high-register version gives each thread more private fast storage, but that storage is paid for out of a fixed SM register file. With `256` threads per block, the high-pressure kernel consumes enough registers that only one block can reside on each SM. That leaves only eight resident warps. The low-pressure version can keep six blocks resident, or forty-eight warps, so the scheduler has many more ready warps available. This is exactly why register count affects occupancy.

Final nuance:

This demo shows that occupancy is a constraint, not a goal by itself. The best kernel is not automatically the one with maximum occupancy. The best kernel is the one with the best runtime after balancing register reuse, spilling, memory traffic, instruction count, and available parallelism.
