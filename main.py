import socket
from collections import deque
from os import path

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
agent = Agent(state_size=144, action_size=6, seed=42)

if path.exists(NAME + '.pth'):
    print("load network from " + NAME + ".pth")
    agent.qnetwork_local.load_state_dict(torch.load(NAME + '.pth'))

max_t = 500
eps_start = 1
eps_end = 0.01
eps_decay = 0.95

scores = []
scores_window = deque(maxlen=100)
eps = eps_start
ep_count = 0

ended = False
while True:
    # data = sock.recv(1024)
    # print("> ", data)
    # # if "S:" not in data.decode("utf-8"):
    # #     continue

    score = 0
    ep_count += 1

    while True:
        data = sock.recv(1024).decode("utf-8").replace('\n', '')
        # print(">> ", data)
        if "M:" in data:
            stateStr = data.split(':')[1].split('/')
            state = np.asarray(stateStr, dtype=np.float64, order='C')
            # print("size: " + str(len(state)))
            # print(state)
            # print("eps: " + str(eps))
            if IS_TRAINING:
                action = agent.act(state, eps)
            else:
                action = agent.act(state, 0)
            # print("action: " + str(action))
            sock.sendall(str.encode(str(action) + "\n"))
        elif "R:" in data:
            reward = int(data.split(':')[1])
            next_state = np.asarray(data.split(':')[3].split('/'), dtype=np.float64, order='C')
            done = data.split(':')[2].lower() in ['true', '1', 'yes']
            # print(reward, next_state, done)
            if IS_TRAINING:
                agent.step(state, action, reward, next_state, done)
            score += reward
            state = next_state
            sock.sendall(str.encode("Updated\n"))
            if done:
                # print("Game ")
                break

        elif "C:" in data:
            sock.close()
            ended = True
            break

    if ended:
        torch.save(agent.qnetwork_local.state_dict(), NAME + '.pth')
        break
    scores.append(score)
    scores_window.append(score)
    eps = max(eps_end, eps * eps_decay)
    mean_score = np.mean(scores_window)

    print(f"\rEpisode: {ep_count}, Eps: {eps:.5f}, Score: {score:.2f}, Average score: {mean_score:.2f}", end="\n")

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
print("ended agent")
print(scores)
