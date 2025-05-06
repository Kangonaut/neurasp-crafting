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

    ```console
    uv sync --extra cpu     # if you simply want to use your CPU
    uv sync --extra cu118   # if you want to use CUDA 11.8
    uv sync --extra cu124   # if you want to use CUDA 12.4
    uv sync --extra cu126   # if you want to use CUDA 12.6
    ```

1. Otherwise, simply use your package manager of choice. E.g. `pip`:

    ```console
    pip install -r pyproject.toml
    ```

> [!WARNING]
> In case you are NOT using `uv`, you need to install `torch` and `torchvision` manually.

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

## Find Stable Models

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
