import sys 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import time
import csv
import random
import jax.numpy as jnp
import jax
from jax import grad
import numpy as np
from itertools import product

# Random Integer with Jax
def random_float(key, low=2, high=50, shape=()):
    key, subkey = jax.random.split(key)
    return key, jax.random.uniform(subkey, shape=shape, minval=low, maxval=high)

# creates a graph
def make_graph(iter, data, color="blue", thickness=0.05):
    x = jnp.linspace(0, iter, len(data))
    y = data
    plt.plot(x, y, color=color, alpha=thickness, marker=None)
    return

#The Four Functions used in Dr. Webb's Paper
def ackley(x):
    n = x.shape[0]
    return (-20 * jnp.exp(-0.2 * jnp.sqrt(jnp.sum(x**2) / n))
            - (jnp.exp(jnp.sum(jnp.cos(2*jnp.pi*x)) / n)) 
            + 20
            + jnp.exp(1))

def rastrigin(x):
    return (10 * x.shape[0]) + jnp.sum(x**2 - 10*jnp.cos(2*jnp.pi*x))  

def rosenbrock(x):
    return jnp.sum(100*( x[1:] - x[:-1]**2)**2 + (1-x[:-1])**2)

def zakharov(x):
    i = jnp.arange(1, x.shape[0] + 1)
    inner = jnp.sum(0.5 * i * x)
    return jnp.sum(x**2) + inner**2 + inner**4

def adam(grad, theta, t_d_theta, m, v, k, n=0.01, eps=10**-8, beta_1=0.9, beta_2=0.99):
    if t_d_theta == None:
        g = grad(theta)
    else:
        temp = grad(t_d_theta)
        g = jnp.array([temp[i,i] for i in range(len(temp))])
    m = beta_1 * m + (1-beta_1)*g
    v = beta_2 * v + (1-beta_2)*(g**2)
    m_hat = m / (1 - beta_1**k)
    v_hat = v / (1 - beta_2**k)
    v_hat = jnp.maximum(v_hat, eps) # Prevents dividing by zero during startup
    update = theta - (n * m_hat) / (jnp.sqrt(v_hat) + eps)        
    return update, m, v


def rand_D(L, size):
    ran = lambda: random.randint(0,L)
    D = jnp.array([[ran() for i in range(size)] for i in range(size)])
    return D

def print_theta(theta):
    arr = jnp.array(theta)  # convert from jnp to numpy
    for row in arr.reshape(-1, arr.shape[-1]):
        a = ("  ".join(f"{v:.4f}" for v in row))
    return a


def print_info(func, start_coord, theta, iter, D, flag, start_time):
    # Get the total time of the function call
    end = time.time()
    total_time = end - start_time

    # Print Data to the Terminal
    a = print_theta(theta)

    if flag == "-a":
        print(f"Equation: Ackley")
    if flag == "-ra":
        print(f"Equation: Rastrigin")
    if flag == "-ro":
        print(f"Equation: Rosenbrock")
    if flag == "-z":
        print(f"Equation: Zakharov")

    
    print(f"Starting point: {start_coord}")
    print(f"Time Delay Matrix:\n{D}")
    print(f"Execution time: {total_time:.4f} seconds")
    print(f"The local minimum is ({a}) with magnitude of {func(theta):.6f}")
    print(f"# of iterations: {iter}\n")


def workflow(coordinate, flag="-ro", d_matrix=None): 
    # Inital condition/ Set up
    # Function flags (-a, -ra, -ro, -z)
    if flag == "-a":
        func = ackley
    elif flag == "-ra":
        func = rastrigin
    elif flag == "-ro":
        func = rosenbrock
    elif flag == "-z":
        func = zakharov
    else:
        raise TypeError(f"{flag} is not an accepted input")
    
    a = len(coordinate)
    theta = jnp.array([float(coordinate[i]) for i in range(a)])
    m = jnp.zeros(a)
    v = jnp.zeros(a)

    # The time-delayed matrix 'd_matrix', and max time delay L
    if d_matrix == None:
        d_matrix = jnp.zeros((a, a))
    L = jnp.max(d_matrix)
 
    iter = 1

    # time tracking for the main function
    start = time.time()
 
    while (iter<10000  and func(theta)>0.01):

        if iter == 1: # trace of all x_i values.
            lst_all_theta = [[float(theta[i])] for i in range(a)]
            outputs = [float(func(theta))]
        elif iter > 1:
            for i in range(a):
                lst_all_theta[i].append(float(theta[i]))
            outputs.append(float(func(theta)))
        g = grad(func)
        if len(outputs) < L + 1 or L == 0:
            theta, m, v = adam(g, theta, None, m, v, iter)
        else:
            # D = rand_D(1, a)
            t_delay_theta = jnp.array([
                [lst_all_theta[x][int(-d_matrix[x, y] - 1)] for y in range(a)] 
                for x in range(a)
            ])
            theta, m, v = adam(g, theta, t_delay_theta, m, v, iter)
        iter += 1
    
    print_info(func, coordinate, theta, iter, d_matrix, flag, start)
    if L == 0:
        make_graph(iter, outputs, color="red", thickness=0.3)
    make_graph(iter, outputs, color="blue")
    return iter


def main():
    # Define the function
    flag = "-ro"
    iter_avg = []

    # Key is the seed for random numbers
    key = jax.random.PRNGKey(0)
    for i in range(2, 3):

        # Make every possible matrix of dim i by i
        matrices = jnp.array(list(product([0, 1], repeat=(i*i))), dtype=jnp.int8).reshape(-1, i, i)

        # Collect 20 Random points of size i
        x_noughts = []
        for j in range(1):
            key, x = random_float(key, shape=(i,))
            x_noughts.append(tuple(x.tolist()))

        for matrix in matrices:
            data = []
            for x in x_noughts:
                out = workflow(x, flag, d_matrix=matrix)
                data.append(out)
            iter_avg.append((sum(data) / len(x_noughts), matrix))

        sorted_avg = sorted(iter_avg, key=lambda x: x[0])
        with open(f"{i}_by_{i}_matrices", "w") as file:
            for t in sorted_avg:
                iter, d_matrix = t
                file.write(f'{d_matrix}: {iter}\n')

        # Save all of the graphs together
        plt.xlabel("x")
        plt.ylabel("y (log scale)")
        plt.yscale('log')
        plt.title(f"Rosenbrock with {i} by {i} time delays")
        plt.savefig(f"{i}_by_{i}_matrices graph.png")
        plt.close()


if __name__ == "__main__":
    main()
    