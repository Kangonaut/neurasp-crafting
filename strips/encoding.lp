% action: cut wood
action(cut_wood).
pre(cut_wood,wood).
pre(cut_wood,axe).
add(cut_wood,wood_plank).
del(cut_wood,wood).

% action: wait
action(wait).

% action: grow sapling
action(grow_sapling).
pre(grow_sapling,water).
pre(grow_sapling,sapling).
pre(grow_sapling,sunlight).
add(grow_sapling,plant).
del(grow_sapling,sapling).

% action: grind coffee beans
action(grind_coffee_beans).
pre(grind_coffee_beans,coffee_beans).
pre(grind_coffee_beans,mortar).
add(grind_coffee_beans,coffee_powder).
del(grind_coffee_beans,coffee_beans).

% action: make coffee
action(make_coffee).
pre(make_coffee,water).
pre(make_coffee,coffee_powder).
pre(make_coffee,bottle).
add(make_coffee,coffee).
del(make_coffee,coffee_powder).
del(make_coffee,bottle).

% initial inventory
have(X,0) :- init(X).

% action rules
{ occur(A,T) : action(A) } = 1 :- time(T).
:- occur(A,T), pre(A,I), not have(I,T-1).
have(I,T) :- time(T), have(I,T-1), not occur(A,T) : del(A,I).
have(I,T) :- occur(A,T), add(A,I).

% goal
:- goal(X), not have(X,k).

#show occur/2.
