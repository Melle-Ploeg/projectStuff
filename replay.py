import time

import torch
import cv2

frames = torch.load('atari-agents-main/recordings/episode0_frames.pt')
actions = torch.load('atari-agents-main/recordings/episode0_actions.pt')
rewards = torch.load('atari-agents-main/recordings/episode0_rewards.pt')

# to_numpy = frames.numpy()

for i in range(len(frames)):
    frame = frames[i]
    action = str(int(actions[i]))
    reward = str(int(rewards[i]))
    # print(int(action))
    # print(type(action))
    resized = cv2.resize(frame.numpy(), [420, 420], interpolation=cv2.INTER_LINEAR)
    resized = cv2.putText(resized, action, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                          1, (255, 255, 255), 2)
    resized = cv2.putText(resized, reward, (370, 40), cv2.FONT_HERSHEY_SIMPLEX,
                          1, (255, 255, 255), 2)
    cv2.imshow("yo", resized)
    cv2.waitKey(0)

cv2.destroyAllWindows()

