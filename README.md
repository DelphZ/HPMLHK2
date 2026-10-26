# HW2: Profiling and Accelerating a Small LLM Workload

**COMS 6998, High Performance Machine Learning, Columbia University**

**Instructor:** Dr. Kaoutar El Maghraoui

**Due:** October 25, 2026

**Points:** 100 (+10 extra credit)

## What you will do

You will fine-tune DistilBERT on IMDB with a training loop you write yourself,
then profile it and make it faster. By the end you should be able to:

- Write a PyTorch training loop for a Hugging Face transformer without leaning
  on `transformers.Trainer`, so you can see and time every stage.
- Split a training step into data loading, forward, backward, and optimizer,
  and time each one correctly on a GPU.
- Look at a step and say whether it is launch-bound, GPU-compute-bound, or
  input-bound, and back that up with trace evidence and a batch-size scaling
  test.
- Tune the input pipeline (`num_workers`, `pin_memory`, `non_blocking`) and show
  whether the host-to-device copy overlaps compute.
- Read a `torch.profiler` trace in Perfetto and use it to find the real
  bottleneck.
- Compile the model, find and fix a graph break, and work out whether compiling
  is worth it for a job of a given length.
- Turn on mixed precision and measure what it does to step time and memory.
- Compare SGD, Adam, and AdamW on speed, convergence, and memory.

## Setup

You will develop against Python 3.11+, PyTorch 2.9, and CUDA 12.x. The
autograded functions and the short-answer checks run on CPU against a tiny
model and a 300-row IMDB sample that ships with the starter, so you can write
and test those parts on your laptop with no GPU.

The real experiments need a GPU: the full IMDB dataset, real DistilBERT, the
profiler traces, the sweeps, compile, and mixed precision. Run those on Google
Colab or the Columbia Insomnia cluster. Keep each profiled or swept run to a
short slice, a few hundred steps, so the whole notebook finishes in one GPU
session.

Weights and Biases is optional. If you use it, run it offline. Grading never
touches the network.

Put every table, number, and paragraph of analysis in the notebook itself, in
markdown cells right under the task it belongs to. The sample tables below show
the columns we expect. Fill them with your own numbers; the exact values depend
on your GPU.

## Coding tasks

### C1 (12 pts): Fine-tune a small LLM with your own loop

Load IMDB, tokenize with the matching Hugging Face tokenizer, and fine-tune
DistilBERT for sentiment classification with a training loop you write around a
PyTorch `DataLoader`. Report training loss and accuracy per epoch. Your code
should run on both the tiny CPU sample and the full GPU dataset.

Autograded, both in the notebook as stubs you fill in:
`make_collate_fn(tokenizer)` returns a `collate_fn(batch)` that pads each
example to the longest in the batch, builds the attention mask, and stacks the
labels; `train_one_step(model, batch, optimizer, device)` runs one step and
returns the loss as a Python float.

Hand in your two functions plus a per-epoch table:

| Epoch | Train loss | Train accuracy |
|---|---|---|
| 1 | ... | ... |
| 2 | ... | ... |

### C2 (8 pts): Baseline timing breakdown

For a short slice, measure and report per step and per epoch: data-loading
time, forward, backward, optimizer, and total. Put `torch.cuda.synchronize()`
around the GPU-timed regions, and say in a sentence or two why timing GPU work
without a sync gives you the wrong answer.

Autograded: `time_training_step(model, batch, optimizer, device)` returns a
dict with keys `data`, `forward`, `backward`, `optimizer`, and `total`, each a
float number of seconds.

Hand in a timing table and the sync explanation:

| Epoch | Data (s) | Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---|---|---|---|---|---|
| 1 | ... | ... | ... | ... | ... |

### C3 (10 pts): Input pipeline tuning

You write the sweep yourself. Fill in the stub
`sweep_num_workers(rows, worker_counts=(0, 2, 4, 8), batch_size=16)`, which
builds a `DataLoader` at each worker count, runs one epoch, and returns a dict
mapping worker count to epoch time in seconds.

a) Sweep `num_workers` over {0, 2, 4, 8}. Plot epoch time and data-loading time
against the worker count, pick the best setting, and explain the curve: why 0
is slow and why the gains flatten out.

b) At your best worker count, compare `pin_memory=False` against
`pin_memory=True` with a `non_blocking=True` host-to-device copy. From a short
trace, show whether the copy overlaps compute, and say whether pinning actually
helped here and why.

Hand in the plot, plus a small comparison table:

| num_workers | Data-loading time (s) | Epoch time (s) |
|---|---|---|
| 0 | ... | ... |
| 2 | ... | ... |
| 4 | ... | ... |
| 8 | ... | ... |

| Config at best num_workers | Epoch time (s) | Copy overlaps compute? |
|---|---|---|
| pin_memory off | ... | ... |
| pin_memory on, non_blocking | ... | ... |

### C4 (15 pts): Find the bottleneck with the profiler

Profile a short slice with `torch.profiler`:

a) Use a real profiler schedule (`wait`, `warmup`, `active`) and wrap the
forward, backward, and optimizer regions in `record_function` labels so the
trace is readable.

b) Open the trace in Perfetto. Include the operator table and the labeled
timeline (screenshots), and report the time split across data, forward,
backward, and optimizer, plus the kernel vs CPU-exec split.

c) Say whether the step is launch-bound, GPU-compute-bound, or input-bound.
Back it with two independent pieces of evidence: a batch-size scaling test and
at least two features you can point to in the trace (gaps between kernels, a
long sync bar, the GPU sitting idle waiting for the next batch, or a run of tiny
kernels). For the scaling test, fill in the stub
`batch_scaling_test(sizes=(8, 16, 32, 64))`, which times one step at each batch
size and returns a dict mapping batch size to per-step seconds. If per-step time
barely moves as the batch grows, the step is launch-bound.

d) Name the first change you would make, and predict what it does.

Autograded, two stubs: `make_profiler_schedule()` returns a
`torch.profiler.schedule` with a nonzero wait, warmup, and active phase;
`profiled_step(model, batch, optimizer, device)` wraps the three regions in
`record_function("forward")`, `record_function("backward")`, and
`record_function("optimizer")`, and returns the loss as a float.

Hand in the two Perfetto screenshots, your batch-scaling table, and a short
paragraph with the regime call, the two trace features, and your first change:

| Batch size | Per-step time (ms) |
|---|---|
| 8 | ... |
| 16 | ... |
| 32 | ... |
| 64 | ... |

### C5 (10 pts): Graph breaks

Compile the model with `torch.compile`, then:

a) Report how many graph breaks you get and what caused them, using
`torch._dynamo.explain` (or `TORCH_LOGS="graph_breaks"`).

b) Fix at least one break (say, a data-dependent branch or a Python operation
on tensor values) and show the count drop.

c) Show the before and after in a trace, pointing to where the break shows up
as a return to Python between compiled regions.

Autograded: the notebook cell sets `gb_before` and `gb_after` to the
graph-break counts (integers) before and after your fix, with `gb_after`
smaller than `gb_before`.

Hand in the counts, the cause of the break you fixed, and the before/after
trace screenshots:

| | Graph breaks | What caused them |
|---|---|---|
| before | ... | ... |
| after | ... | ... |

### C6 (12 pts): torch.compile and when it pays off

You write the timing helper. Fill in the stub
`avg_step_time(model, batch, optimizer, device, iters=10)`, which returns the
average seconds per step over `iters` steps after a short warm-up, with the GPU
synchronized around the timed region.

a) Measure the average step time for eager and for compiled (default Inductor)
after a proper warm-up, and report the speedup.

b) Measure the one-time compile cost (the cold start). Work out the break-even
step count: how many steps the job has to run before compiled, compile cost
included, comes out ahead of eager. Say whether compiling is worth it for a
20-step job and for a full training run, and show your arithmetic.

c) Try one non-default mode, `reduce-overhead` or `max-autotune`, and say
whether it helped here.

Hand in a table and the break-even arithmetic:

| Mode | Avg step time (ms) | Compile cost (s) | Break-even (steps) |
|---|---|---|---|
| eager | ... | n/a | n/a |
| compiled, default | ... | ... | ... |
| compiled, reduce-overhead or max-autotune | ... | ... | ... |

### C7 (10 pts): Mixed precision

Add mixed precision to your step with `torch.autocast` and a `GradScaler`.
Against the fp32 baseline, report the change in step time and in peak GPU memory
(`torch.cuda.max_memory_allocated`). Say which operations run in lower precision
and which stay in fp32, and what turning on TF32 matmuls would and would not
change.

Autograded: `mixed_precision_step(model, batch, optimizer, device, scaler=None)`
runs the forward under a `torch.autocast` block and returns the loss as a float.

Hand in a table and the short explanation:

| | Avg step time (ms) | Peak memory (MB) |
|---|---|---|
| fp32 | ... | ... |
| mixed precision | ... | ... |

### C8 (8 pts): Optimizers on time, convergence, and memory

Compare SGD, Adam, and AdamW. For each, report the average epoch time and how
it converges. Then:

a) Work out by hand how many bytes of optimizer state each one keeps for this
model (none, one, and two tensors per parameter), and check the ordering
against a memory profile or `max_memory_allocated`.

b) Compare them by wall-clock time to reach a fixed target loss, not by a fixed
epoch count, and say which gets there first.

Autograded: `build_optimizer(model, name, lr)` for `name` in `"sgd"`, `"adam"`,
`"adamw"`; and `optimizer_state_bytes(num_params, name, momentum=0.0)`, which
returns the fp32 optimizer-state bytes (SGD 0, SGD with momentum
`4 * num_params`, Adam and AdamW `8 * num_params`).

Hand in a table:

| Optimizer | Epoch time (s) | Final loss | Optimizer-state bytes | Time to target loss (s) |
|---|---|---|---|---|
| SGD | ... | ... | ... | ... |
| Adam | ... | ... | ... | ... |
| AdamW | ... | ... | ... | ... |

## Short-answer questions (15 pts)

Answer each in a labeled markdown cell in the notebook.

- Q1 (2 pts): the input dimension of DistilBERT's embedding layer. (autograded)
- Q2 (2 pts): the output dimension of the classifier head for IMDB. (autograded)
- Q3 (4 pts): the number of trainable parameters, and the number carrying
  gradients after a backward pass; include your code. (autograded)
- Q4 (3 pts): from your C8 figures, explain how optimizer memory changes from
  SGD to Adam, in bytes per parameter.
- Q5 (2 pts): from your C6 numbers, state in one sentence when compiling pays
  off.
- Q6 (2 pts): name the bands in a GPU memory timeline (parameters, gradients,
  optimizer state, activations, temporaries) and say which one grows fastest
  with sequence length.

## Extra credit (+10)

### EC1 (+5): hunt down a hidden sync

Find a hidden CPU-GPU synchronization in a step, for example an `.item()`, a
`.cpu()`, or a Python comparison on a tensor value inside the loop. Show its
cost as a sync bar in the trace, remove or defer it, and show the step time
drop. Hand in the before/after step time and the trace screenshots.

### EC2 (+5): recompilation and dynamic shapes

Dynamic padding gives you variable sequence lengths, which trigger
recompilation. Log the recompiles across a range of batch or sequence sizes,
then mark the varying dimension dynamic (`torch._dynamo.mark_dynamic` or
`dynamic=True`) and show the recompile count fall. Say which guard was failing.
Hand in the recompile counts before and after.

## Submitting

Work in the starter notebook, `master.ipynb`. Each function you have to write
is a stub with a prompt right above it; fill in the body where it says `...` and
keep the function name and signature exactly as given. Do not rename anything or
start a fresh notebook, the autograder looks your functions up by name.

Submit that same notebook to Gradescope. It has to run top to bottom from a
fresh kernel with no errors. Keep every table, plot, screenshot, and paragraph
of analysis in the notebook, under the task it belongs to.

## How grading works

23 of the 100 points are autograded on CPU from the functions you fill in:
`make_collate_fn`, `train_one_step`, `time_training_step`,
`make_profiler_schedule`, `profiled_step`, `mixed_precision_step`,
`build_optimizer`, `optimizer_state_bytes`, the `gb_before`/`gb_after` counts,
the embedding and classifier dimension answers, and the parameter counts. Each
task above says exactly what its function takes and returns.

Everything else is graded from what you submit: the GPU experiments, plots, tables,
profiler screenshots, and your written analysis. Passing the public check next
to a function does not mean full credit. A stricter hidden check runs at
grading time.

| Task | Focus | Points | How it is graded |
|---|---|---|---|
| C1 | Fine-tune with your own loop | 12 | autograder + write-up |
| C2 | Baseline timing breakdown | 8 | autograder + write-up |
| C3 | Input pipeline: workers, pinning, overlap | 10 | write-up |
| C4 | Bottleneck diagnosis with evidence | 15 | autograder + write-up |
| C5 | Graph breaks: find, fix, show | 10 | autograder + write-up |
| C6 | Compile performance and break-even | 12 | write-up |
| C7 | Mixed precision | 10 | autograder + write-up |
| C8 | Optimizers: time, memory, time-to-target | 8 | autograder + write-up |
| Q1-Q6 | Short answers | 15 | autograder + write-up |
| EC1 | Hidden sync hunt | +5 | write-up |
| EC2 | Recompilation and dynamic shapes | +5 | write-up |
#   H P M L H K 2  
 