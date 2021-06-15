import socket
from collections import deque
from os import path

import matplotlib
import numpy as np
import torch
import matplotlib.pyplot as plt
from dqn_agent import Agent

HOST = "localhost"
PORT = 8080

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))
WELCOME_PACKET = sock.recv(1024).decode("utf-8").replace('\n', '')
print("Welcome > " + WELCOME_PACKET)
NAME = WELCOME_PACKET.split(":")[0]

print(WELCOME_PACKET.split(":")[1].lower())
IS_TRAINING = WELCOME_PACKET.split(":")[1].lower() in ['true', '1', 'yes']

print("Name:" + NAME)
print("Is Training: " + str(IS_TRAINING))
agent = Agent(state_size=196, action_size=6, seed=42)

if path.exists(NAME + '.pth'):
    print("load network from " + NAME + ".pth")
    agent.qnetwork_local.load_state_dict(torch.load(NAME + '.pth'))

total_generation = 10
training_episode = 3000
testing_episode = 100

max_t = 500

scores_window = deque(maxlen=100)
ep_count = 0
ended = False

isTraining = True
generation = 0
episode = 0
py_generation = 1

scores = []
training_scores = []
eval_scores = []
eps_start = 0.3
eps_end = 0.01
eps_decay = 0.99

while True:

    eps = eps_start

    while True:
        score = 0
        while True:
            data = sock.recv(1024).decode("utf-8").replace('\n', '')
            # print(">> ", data, end="\n")
            if "M:" in data:
                stateStr = data.split(':')[1].split('/')
                state = np.asarray(stateStr, dtype=np.float64, order='C')
                # print("size: " + str(len(state)))
                # print(state)
                # print("eps: " + str(eps))
                if isTraining:
                    action = agent.act(state, eps)
                else:
                    action = agent.act(state, -1)
                # print("action: " + str(action))
                sock.sendall(str.encode(str(action) + "\n"))
            elif "R:" in data:
                reward = int(data.split(':')[1])
                done = data.split(':')[2].lower() in ['true', '1', 'yes']
                next_state = np.asarray(data.split(':')[3].split('/'), dtype=np.float64, order='C')
                # print(reward)
                if isTraining:
                    agent.step(state, action, reward, next_state, done)
                score += reward
                state = next_state
                sock.sendall(str.encode("Updated\n"))
                # if done:
                #     # print("Game ")
                #     break
            elif "ST:" in data:
                isTraining = data.split(':')[1].lower() in ['true', '1', 'yes']
                generation = int(data.split(':')[2])
                episode = int(data.split(':')[3])
                sock.sendall(str.encode("Started\n"))
                # print("Started new game", eps, eps*eps_decay, eps_end)
                eps = max(eps_end, eps*eps_decay)
                break

            elif "C:" in data:
                sock.close()
                ended = True
                break

        if ended:
            torch.save(agent.qnetwork_local.state_dict(), NAME + '.pth')
            break
        scores.append(score)

        if isTraining:
            training_scores.append(score)
        else:
            eval_scores.append(score)

        scores_window.append(score)
        mean_score = np.mean(scores_window)
        # if ep_count % 100 == 0:
        if not isTraining:
            print(
                f"\rGen: {generation}, Ep: {episode}, T: {isTraining}, Eps: {eps:.5f}, Score: {score:.2f}, Mean: {mean_score:.2f}",
                end="\n")

        if generation != py_generation:
            py_generation = generation
            break
    if ended:
        break

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

plt.plot(np.arange(len(training_scores)), training_scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

ax = fig.add_subplot(111)
plt.plot(np.arange(len(eval_scores)), eval_scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
