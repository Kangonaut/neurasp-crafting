# NeurASP Crafting

## Credit

- original NeurASP paper: [NeurASP: Embracing Neural Networks into Answer Set Programming](https://arxiv.org/abs/2307.07700) by **Zhun Yang, Adam Ishay and Joohyung Lee**
- [original implementation](https://github.com/azreasoners/NeurASP) of the NeurASP architecture
- this project makes use of the [icon pack](https://shikashipx.itch.io/shikashis-fantasy-icons-pack) by [shikashipx](https://itch.io/profile/shikashipx)

## Setup

1. Clone this repo using

    ```console
    git clone https://github.com/Kangonaut/neurasp-crafting.git
    ```

1. If you are using [`uv`](https://github.com/astral-sh/uv), simply install all dependencies using
    - **NOTE:** Since the CNN used in this example is very small, it does not make a huge difference if you use a GPU or not.

    ```console
    uv sync --extra cpu     # if you simply want to use your CPU
    uv sync --extra cu118   # if you want to use CUDA 11.8
    uv sync --extra cu124   # if you want to use CUDA 12.4
    uv sync --extra cu126   # if you want to use CUDA 12.6
    ```

1. Otherwise, simply use your package manager of choice. E.g. `pip`:
    - **NOTE:** In case you are NOT using `uv`, you need to install `torch` and `torchvision` manually.

    ```console
    pip install -r pyproject.toml
    ```

1. Download the icon pack [here](https://shikashipx.itch.io/shikashis-fantasy-icons-pack).
1. Extract the ZIP file and rename the `BG 10.png` file to `sprites.png` and save it in the project's root folder.
1. Generate the individual sprites using

    ```console
    uv run generate-sprites.py
    ```

1. Next, generate the dataset using

    ```console
    uv run generate-dataset.py
    ```

1. Finally, training!!! :D

    ```console
    uv run train-neurasp.py
    ```

## Configuration

The `strips.yml` file defines the configuration parameters for the task that the model is supposed to solve.

- `time_steps`: each action requires one time steps, so this parameter defines the amount of actions that are applied between the initial and final inventory state
- `inventory_size`: the maximum amount of initial items in the inventory
- `items`: the available items
  - `id`: a unique identifier of the item
    - **NOTE:** `0` is a reserved identifier for the blank item (i.e. no item)
  - `name`: a human readable identifier
  - `path`: the path to the item sprite
- `actions:` the available actions
  - `name`: a unique identifier of the action
    - **NOTE:** must only consist of upper case letters, lower case letters and the underscore `_` character
  - `preconditions`: the items needed to apply this action
  - `add_list`: the items that are created when applying this action
  - `delete_list`: the items that are consumed/removed when applying this action

```yml
time_steps: 2
inventory_size: 4
items:
  - id: 1
    name: coffee
    path: ./sprites/15-14.png
  - id: 2
    name: coffee_powder
    path: ./sprites/15-11.png
  - id: 3
    name: water
    path: ./sprites/19-5.png
  - id: 4
    name: milk
    path: ./sprites/15-8.png
  - id: 5
    name: flour
    path: ./sprites/15-10.png
  - id: 6
    name: bread
    path: ./sprites/14-13.png
actions:
  - name: make_coffee
    preconditions:
      - coffee_powder
      - water
    add_list:
      - coffee
    delete_list:
      - coffee_powder
      - water
  - name: bake_bread
    preconditions:
      - water
      - flour
    add_list:
      - bread
    delete_list:
      - water
      - flour
  - name: wait
    preconditions: []
    add_list: []
    delete_list: []
```

## Training

For training, there are two options. Firstly, you can train the simply CNN network, which means there is no logic layer involved, just plain old deep learning. You may want to this, if you want to test the NeurASP model with a pre-trained neural network. The model weights will be stored in `./results/network/train-RUN/`, where `RUN` will be replaced by the next run number (starting with 0). The script saves both the best model weights (based on the validation accuracy) as `best.pt` and final models weights after the entire training run as `last.pt`.

```console
$ uv run train-network.py
[TRAIN] loss: 0.0508: 100%|█████████████| 1000/1000 [00:03<00:00, 287.56it/s]
[TRAIN] epoch 1:
        loss: 0.1187
        accuracy: 83.20 %
[VALID]: loss: 0.0022: 100%|██████████████| 200/200 [00:00<00:00, 537.57it/s]
[VALID] epoch 1:
        loss: 0.0354
        accuracy: 96.75 %
[TEST]: loss: 0.0528: 100%|███████████████| 200/200 [00:00<00:00, 459.89it/s]
test loss: 0.0344
test accuracy: 96.75 %
```

The second (and more interesting) option is to train the NeurASP model. This means that you will start off with a fresh CNN model that predicts the probability distributions of the items given the input images, that will then be adjusted by the logic layer to improve the probability of *likely* (based on the number of different stable models that satisfy a certain situation) predictions and reduce the probability of *unlikely* predictions. The weights of the CNN model(s) are stored in `./results/neurasp/train-RUN`, where again `RUN` will be replaced by the corresponding run number. Also, we again store the `best.pt` and `last.pt` model weights.

```console
$ uv run train-neurasp.py
validation accuracy before training: 0.00 %
epoch 1/1:
100%|██████████████████████████████████████| 500/500 [06:57<00:00,  1.20it/s]
        training accuracy: 52.00 %
        validation accuracy: 45.00 %
NeurASP test accuracy: 50.00 %
CNN test accuracy: 50.00 %
```

> [!TIP]
> **Experiencing lengthy training durations?** Since training requires computing all the stable models for each sample in the dataset, it is a very intensive and CPU heavy task. So it takes a lot of time and cannot be sped up with GPU utilization. I recommend keeping the amount of items and actions small, in order to reduce the search space for stable models.

## Testing

You can either test the CNN or the entire NeurASP model. For the former, you first need to look up the path to the model weights. If you trained your model using the `train-network.py` script, this will be similar to `./results/network/train-0/best.pt`. If you trained your model using the `train-neurasp.py` script, it will instead be something like `./results/neurasp/train-0/identify/best.pt`, where `identify` is the unique identifier for the neural network in the NeurASP model. Now, simply pass this path as the `--model` argument to the `test-network.py` script.

```console
$ uv run test-network.py --model ./results/neurasp/train-1/identify/best.pt
[TEST]: loss: 0.0153: 100%|███████████████| 200/200 [00:00<00:00, 344.28it/s]
test loss: 0.0338
test accuracy: 95.50 %
```

If you want to test the entire NeurASP model, simply use the `test-neurasp.py` script and pass the model path as the `--model` argument argument. Note that the model path should point to point to the directory that contains all of the neural network weights. For example `./results/neurasp/train-0/`.

```console
$ uv run test-neurasp.py --model ./results/neurasp/train-1/
NeurASP test accuracy: 71.00 %
CNN test accuracy: 71.00 %
```

## Things to Try

### Generate the ASP Program

Once you have modified the config to your liking (e.g. by adding a few items and actions), you can generate the ASP program using:

```console
$ uv run generate-program.py
time(1..2).
have(X,0) :- init(X).
{ occur(A,T) : action(A) } = 1 :- time(T).
:- occur(A,T), pre(A,I), not have(I,T-1).
have(I,T) :- time(T), have(I,T-1), not occur(A,T) : del(A,I).
have(I,T) :- occur(A,T), add(A,I).
item(0).
item(1).
item(2).
item(3).
item(4).
item(5).
item(6).
action(make_coffee).
pre(make_coffee,2).
pre(make_coffee,3).
add(make_coffee,1).
del(make_coffee,2).
del(make_coffee,3).
action(bake_bread).
pre(bake_bread,3).
pre(bake_bread,5).
add(bake_bread,6).
del(bake_bread,3).
del(bake_bread,5).
action(wait).
```

### Find Stable Models

Let's say, you want to check what things you can craft with *coffee beans* *water* and a *mortar*.
Simply generate the ASP program with the initial inventory and then pass it on to `clingo`:

```console
$ uv run generate-program.py --init coffee_beans water mortar | clingo 0
clingo version 5.8.0 (6d1efb6)
Reading from stdin
Solving...
Answer: 1 (Time: 1.090s)
time(1) time(2) init(2) init(5) init(4) have(2,0) have(5,0) have(4,0) have(4,1) have(4,2) action(grind_coffee_beans) action(make_coffee) action(grind_corn) action(bake_cake) action(bake_bread) action(wait) pre(grind_coffee_beans,4) pre(grind_coffee_beans,2) pre(make_coffee,3) pre(make_coffee,5) pre(grind_corn,7) pre(grind_corn,4) pre(bake_cake,8) pre(bake_cake,6) pre(bake_cake,9) pre(bake_bread,5) pre(bake_bread,9) del(grind_coffee_beans,2) del(make_coffee,3) del(make_coffee,5) del(grind_corn,7) del(bake_cake,8) del(bake_cake,6) del(bake_cake,9) del(bake_bread,5) del(bake_bread,9) add(grind_coffee_beans,3) add(make_coffee,1) add(grind_corn,9) add(bake_cake,10) add(bake_bread,11) item(0) item(1) item(2) item(3) item(4) item(5) item(6) item(7) item(8) item(9) item(10) item(11) have(5,1) have(2,1) have(2,2) have(5,2) occur(wait,1) occur(wait,2)
Answer: 2 (Time: 1.090s)
time(1) time(2) init(2) init(5) init(4) have(2,0) have(5,0) have(4,0) have(4,1) have(4,2) action(grind_coffee_beans) action(make_coffee) action(grind_corn) action(bake_cake) action(bake_bread) action(wait) pre(grind_coffee_beans,4) pre(grind_coffee_beans,2) pre(make_coffee,3) pre(make_coffee,5) pre(grind_corn,7) pre(grind_corn,4) pre(bake_cake,8) pre(bake_cake,6) pre(bake_cake,9) pre(bake_bread,5) pre(bake_bread,9) del(grind_coffee_beans,2) del(make_coffee,3) del(make_coffee,5) del(grind_corn,7) del(bake_cake,8) del(bake_cake,6) del(bake_cake,9) del(bake_bread,5) del(bake_bread,9) add(grind_coffee_beans,3) add(make_coffee,1) add(grind_corn,9) add(bake_cake,10) add(bake_bread,11) item(0) item(1) item(2) item(3) item(4) item(5) item(6) item(7) item(8) item(9) item(10) item(11) have(5,1) have(3,2) occur(grind_coffee_beans,2) have(2,1) have(5,2) occur(wait,1)
Answer: 3 (Time: 1.090s)
time(1) time(2) init(2) init(5) init(4) have(2,0) have(5,0) have(4,0) have(4,1) have(4,2) action(grind_coffee_beans) action(make_coffee) action(grind_corn) action(bake_cake) action(bake_bread) action(wait) pre(grind_coffee_beans,4) pre(grind_coffee_beans,2) pre(make_coffee,3) pre(make_coffee,5) pre(grind_corn,7) pre(grind_corn,4) pre(bake_cake,8) pre(bake_cake,6) pre(bake_cake,9) pre(bake_bread,5) pre(bake_bread,9) del(grind_coffee_beans,2) del(make_coffee,3) del(make_coffee,5) del(grind_corn,7) del(bake_cake,8) del(bake_cake,6) del(bake_cake,9) del(bake_bread,5) del(bake_bread,9) add(grind_coffee_beans,3) add(make_coffee,1) add(grind_corn,9) add(bake_cake,10) add(bake_bread,11) item(0) item(1) item(2) item(3) item(4) item(5) item(6) item(7) item(8) item(9) item(10) item(11) have(5,1) have(3,1) occur(grind_coffee_beans,1) have(3,2) have(5,2) occur(wait,2)
Answer: 4 (Time: 1.090s)
time(1) time(2) init(2) init(5) init(4) have(2,0) have(5,0) have(4,0) have(4,1) have(4,2) action(grind_coffee_beans) action(make_coffee) action(grind_corn) action(bake_cake) action(bake_bread) action(wait) pre(grind_coffee_beans,4) pre(grind_coffee_beans,2) pre(make_coffee,3) pre(make_coffee,5) pre(grind_corn,7) pre(grind_corn,4) pre(bake_cake,8) pre(bake_cake,6) pre(bake_cake,9) pre(bake_bread,5) pre(bake_bread,9) del(grind_coffee_beans,2) del(make_coffee,3) del(make_coffee,5) del(grind_corn,7) del(bake_cake,8) del(bake_cake,6) del(bake_cake,9) del(bake_bread,5) del(bake_bread,9) add(grind_coffee_beans,3) add(make_coffee,1) add(grind_corn,9) add(bake_cake,10) add(bake_bread,11) item(0) item(1) item(2) item(3) item(4) item(5) item(6) item(7) item(8) item(9) item(10) item(11) have(5,1) have(3,1) occur(grind_coffee_beans,1) have(1,2) occur(make_coffee,2)
SATISFIABLE

Models       : 4
Calls        : 1
Time         : 1.090s (Solving: 0.00s 1st Model: 0.00s Unsat: 0.00s)
CPU Time     : 0.001s
```

So, as we can see, in the first scenario (i.e. the first stable model), we simply wait for two time steps.
In the second and third scenarios, we grind the coffee beans in one time step and wait in the other one.
Finally, in the fourth scenario, we first grind coffee beans and then make our coffee. :D

## Want to learn about ASP and NeurASP?

Here are some useful resources:

- [Answer set solving in practice - a graduate course at the University of Potsdam](https://teaching.potassco.org/)
- [clingo and gringo - ASP grounding and solving](https://potassco.org/clingo/)
- [Neuro Symbolic Reasoning and Learning](https://link.springer.com/book/10.1007/978-3-031-39179-8)
- [NeurASP: Embracing Neural Networks into Answer Set Programming](https://arxiv.org/abs/2307.07700)
