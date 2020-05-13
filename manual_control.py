#!/usr/bin/env python3

import time
import argparse
import numpy as np
from collections import deque
import gym
import gym_minigrid
# from scores.score_logger import ScoreLogger
import matplotlib.pyplot as plt
import random
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window
import logging

def redraw(img):
    if not args.agent_view:
        img = env.render('rgb_array', tile_size=args.tile_size)

    window.show_img(img)

def reset():
    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()

    if hasattr(env, 'mission'):
      #  print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs)

ACTION_IDX = {
    'left' : 0,
    'right' : 1,
    'forward' : 2,
    'pickup' : 3,
    'drop' : 4,
    'toggle' : 5,
    'enter' : 6
}

def step(action):

    print("HIII ", action)

    # TODO: 1. experiment with constant reward of -1 and give 0 on reaching goal.
    #       2. work on the tree to fill the gaps so that it doesn't get stuck.  => to discuss
    #       3. stop the iteration after some max_step
    #       4. Once these works, try the idea of discarding any q-update on reaching max-steps. (cloneTree)
    # run it for 100-200 several runs

    step = 0
    obs, reward, done, info = env.step(action)
    # print('step=%s, reward=%.2f' % (env.step_count, reward))
    logging.info('step=%s, reward=%s', env.step_count, str(reward))

    # new code
    o = obs["image"]
    o = o.transpose()
    # print(o)
    o = preprocess(o)

    input_state = rdf(o)

    #print("***********************input state*********************************** ")
    #print(input_state)
    input_rdf = {}
    for s, p, o in input_state:
        if p in list(input_rdf.keys()):
            input_rdf[p].append(o)
        else:
            # l = [[s, o]]
            input_rdf[p] = [o]
   # print(input_rdf)
    leaf_state = []
    # dTree.root.reward = reward

    # temp_root = dTree.root.cloneTree()

    state = dTree.root.traverse(input_rdf, leaf_state)

    while not done:
        # key_handler.key = action
        brk_flag = window.reg_key_handler(esc_key)
        if brk_flag:
            return
        redraw(obs)

        step += 1

        # get the action at current state
        # print("Action to Take: ", state.assertAction)
        action = state.assertAction

        # get the next state

        obs_next, reward, done, info = env.step(ACTION_IDX[action])
        # print('step=%s, reward=%.2f' % (env.step_count, reward))
        logging.info('step=%s, reward=%s', env.step_count, str(reward))

        # reward = reward if not done else -reward

        o = obs_next["image"]
        o = o.transpose()
        o = preprocess(o)
        input_state = rdf(o)
        # print(input_state)

        input_rdf = {}
        for s, p, o in input_state:
            if p in list(input_rdf.keys()):
                input_rdf[p].append(o)
            else:
                input_rdf[p] = [o]
        # print(input_rdf)

        leaf_state = []         # default
        state.reward = reward

        next_state = dTree.root.traverse(input_rdf, leaf_state)
        # print("Before Q-update: ", next_state.assertAction)
        #
        # D0511
        if (step % 200) == 0 and state.Q_val_list == next_state.Q_val_list:
            dTree.randFlag = True
        dTree.remember(state, ACTION_IDX[action], reward, next_state, done)

        state = next_state
        obs = obs_next

        # break if the number of steps exceed 10k for a Run
        if env.step_count > 10000:
            print('step=%s, reward=%.2f' % (env.step_count, reward))
            print("Not Done!!!")
            RUN_REWARD.append(reward)
            RUN_STEP.append(env.step_count)
            reset()
            break;

        if done:
            print('step=%s, reward=%.2f' % (env.step_count, reward))
            print("done")
            RUN_REWARD.append(reward)
            RUN_STEP.append(env.step_count)
            reset()
            break;
    return step


# Map of object type to integers old
OBJECT_TO_IDX_OLD = {
    'unseen': 0,
    'empty': 1,
    'wall': 2,
    'floor': 3,
    'door': 4,
    'key': 5,
    'ball': 6,
    'box': 7,
    'goal': 8,
    'lava': 9,
    'agent': 10
}

# Map of object type to integers new
OBJECT_TO_IDX_NEW = {
    'agent': 0,
    'key': 1,
    'door': 2,
    'goal': 3
}

def preprocess(old_state):
    dim = len(OBJECT_TO_IDX_NEW)
    old_key_list = list(OBJECT_TO_IDX_OLD.keys())
    old_val_list = list(OBJECT_TO_IDX_OLD.values())
    obs = (3, dim, dim)
    obs = np.zeros(obs, dtype=int)

    visible = (dim, dim)
    visible = np.zeros(visible, dtype=int)

    carrying = (dim, dim)
    carrying = np.zeros(carrying, dtype=int)

    locked = (dim, dim)
    locked = np.zeros(locked, dtype=int)

    for key in OBJECT_TO_IDX_NEW:
        old_index = OBJECT_TO_IDX_OLD[key]
        new_index = OBJECT_TO_IDX_NEW[key]
        found = np.where(old_state[0] == old_index)
        # print("agent ", old_state[0][6][3])
        # print("key ", old_index)

        if (found[0].size > 0):
            visible[0][new_index] = 1

    obs[0] = visible

    if (old_state[0][6][3] != 1):
        carrying_object = old_key_list[old_val_list.index(old_state[0][6][3])]
        new_index = OBJECT_TO_IDX_NEW[carrying_object]
        carrying[0][new_index] = 1

    obs[1] = carrying

    is_door_locked = np.where(old_state[2] == 2)
    door_index = OBJECT_TO_IDX_NEW['door']
    if (is_door_locked[0].size > 0):
        locked[0][door_index] = 1

    obs[2] = locked

    # print(obs)

    return obs


def rdf(o):
    state = []

    visible = o[0]
    carrying = o[1]
    locked = o[2]

    objects_visible = np.where(visible[0] == 1)
    objects_carrying = np.where(carrying[0] == 1)
    door_locked = np.where(locked[0] == 1)

    key_list = list(OBJECT_TO_IDX_NEW.keys())
    val_list = list(OBJECT_TO_IDX_NEW.values())

    # print(locked)

    for b in objects_visible[0]:
        object = key_list[val_list[b]]
        state.append(("agent", "visible", object))

    for b in objects_carrying[0]:
        object = key_list[val_list[b]]
        state.append(("agent", "carrying", object))

    for b in door_locked[0]:
        object = key_list[val_list[b]]
        state.append(("agent", "locked", object))

    # for s, p, o in state:
        # print((s, p, o))
    return state

def esc_key(event):
    if event.key == 'escape':
        # reset()
        window.close()
        return True

def key_handler(event):
    print('pressed', event.key)

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        step(env.actions.left)
        return
    if event.key == 'right':
        step(env.actions.right)
        return
    if event.key == 'up':
        step(env.actions.forward)
        return

    # Spacebar
    if event.key == ' ':
        step(env.actions.toggle)
        return
    if event.key == 'pageup':
        step(env.actions.pickup)
        return
    if event.key == 'pagedown':
        step(env.actions.drop)
        return

    if event.key == 'enter':
        step(env.actions.done)
        return
    if event.key == 'a':
        noOfAttempts = 0
        for i in range (100):
            action = random.randint(0, len(ACTION_TO_IDX)-1)
            t_noOfAttempts = noOfAttempts
            noOfAttempts = noOfAttempts + step(action) + 1
            num_steps = noOfAttempts - t_noOfAttempts + 1
            print("Run #: ", i, "Took Steps: ", num_steps)
            print("average steps to reach the goal ", noOfAttempts/(i+1))
            logging.info('Run #: %s, Took Steps: %s', str(i), str(num_steps))
            logging.info('Average Steps to Reach Goal: %s', str(noOfAttempts / (i+1)))
        return

ACTION_TO_IDX = {
    0 : 'left',
    1 : 'right',
    2 : 'forward',
    3 : 'pickup',
    4 : 'drop',
    5 : 'toggle',
    6 : 'enter'
}

RUN_STEP = []
RUN_REWARD = []
DISCOUNT_FACTOR = 0.9     # D0513
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000
BATCH_SIZE = 20

ROOT_FLAG = False

class Tree:
    def __init__(self):
        self.root = None
        # self.old_state = None   #store the leafnode
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.isFit = False
        self.randFlag = False
        self.cnt = 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        state.q_update(next_state, action, self.randFlag)
        if self.randFlag and self.cnt < 50:
            self.randFlag = False
            self.cnt = 0
        elif self.randFlag:
            self.cnt += 1


class TreeNode:
    def __init__(self, predicate, obj, n=0, assertAction=""):

        self.nodeType = n  ##  test, 1 -> #leaf
        self.parent = None
        self.yes = None
        self.no = None
        self.reward = 0
        self.learning_rate = LEARNING_RATE
        self.discount_fact = DISCOUNT_FACTOR
        self.last_state = []
        self.assertactn = " "

        # for test nodes
        if self.nodeType == 0:
            self.predicate = predicate
            self.obj = obj

        # for leaf nodes
        else:
            self.expression = []
            self.assertAction = assertAction
            self.Q_val = 50
            # TODo
            self.Q_val_list = list(50 for i in range(len(ACTION_TO_IDX)))

    def insert(self, side, val, assertAction=" "):
        # Compare the new value with the parent node
        if len(val) != 0:
            if side == "yes":
                self.yes = TreeNode(val[0], val[1])
                self.yes.parent = self
            else:
                self.no = TreeNode(val[0], val[1])
                self.no.parent = self
        else:
            if side == "yes":
                self.yes = TreeNode("", "", 1, assertAction)
                self.yes.parent = self
            else:
                self.no = TreeNode("", "", 1, assertAction)
                self.yes.parent = self

    def print(self):
        if self.nodeType == 0:
            print("")
            #print("Test node ", self.predicate, self.obj)
        else:
            print("")
            #print("Leaf Node ", self.assertAction, self.expression, self.Q_val)
        if self.yes:
            self.yes.print()
        if self.no:
            self.no.print()

    def q_update(self, next_state, action, flag):
        # print("-------Inside Q-update function--------")
        logging.info("-------Inside Q-update function--------")
        # print("Old State: ", self.expression, " Q-val: ", self.Q_val_list)
        # print("Action : ", ACTION_TO_IDX[action])
        # print("Next State: ", next_state.expression, "Q-val: ", next_state.Q_val_list)

        if flag:
            action = random.randint(0, len(ACTION_TO_IDX) - 1)
            self.assertAction = ACTION_TO_IDX[action]
            return

        # TODO --> calculate the 'estimate of optimal future value'
        #     q_update = reward
        self.Q_val_list[action] = self.Q_val_list[action] + self.learning_rate * (
            self.reward + self.discount_fact * max(next_state.Q_val_list) - self.Q_val_list[action]
        )
        m = max(self.Q_val_list)
        l = [i for i, j in enumerate(self.Q_val_list) if j == m]
        dpstr = str(l).strip('[]')
        logging.info('set of max value action : %s', dpstr)
        # print("set of max value action : ", l)
        # action = self.Q_val_list.index(max(self.Q_val_list))
        if len(l) > 1:
            action = random.choice(l)
        else:
            action = l[0]

        self.Q_val = self.Q_val_list[action]
        self.assertAction = ACTION_TO_IDX[action]

    def get_action(self):
        # todo normalize the q-val, and randomize at periodic time
        if self.assertAction == " ":
            # select random action
            action = random.randint(0, len(ACTION_TO_IDX)-1)
            self.assertAction = ACTION_TO_IDX[action]
            # self.assertactn = self.assertAction

    def traverse(self, predList, state_exp):
        if self.nodeType == 1:
            self.expression = state_exp
            # print("LEAF NODE FOUND, @ State ", self.expression)
            logging.info('LEAF NODE FOUND, @ State: %s', self.expression)
            self.get_action()
            # print("Q-Value State Action Pair: ", self.Q_val, self.assertAction)
            logging.info('Q-Value State Action Pair: %s %s', self.Q_val, self.assertAction)
            return self

        if self.predicate in predList.keys():
            if self.obj in predList[self.predicate]:
                p = self.predicate
                for i in range(len(predList[p])):
                    if predList[p][i] == self.obj:
                        o = predList[p][i]
                if self.yes:
                    state_exp.append([p, o])
                    node = self.yes.traverse(predList, state_exp)
            else:
                if self.no:
                    node = self.no.traverse(predList, state_exp)
        else:
            if self.yes:
                node = self.yes.traverse(predList, state_exp)
            else:
                node = self.no.traverse(predList, state_exp)

        return node

    def cloneTree(self):
        if self.nodeType == 0:
            tempNode = TreeNode(self.predicate, self.obj)
            tempNode.yes = self.yes.cloneTree()
            tempNode.no = self.no.cloneTree()
        else:
            tempNode = TreeNode("", "", self.nodeType, self.assertAction)
            tempNode.expression = self.expression
            tempNode.Q_val_list = self.Q_val_list
            tempNode.Q_val = self.Q_val
        return tempNode

def create_tree():
    dtree = Tree()
    dtree.root = TreeNode("visible", "key")
    # root_node = TreeNode("visible", "key")
    dtree.root.insert("yes", ["carrying", "key"])
    dtree.root.insert("no", [], " ")
    left = dtree.root.yes

    left.insert("yes", ["visible", "door"])
    left.insert("no", [], " ")

    left = left.yes

    left.insert("yes", ["locked", "door"])
    left.insert("no", [], " ")

    left = left.yes

    left.insert("yes", [], " ")
    left.insert("no", ["visible", "door"])

    right = left.no

    right.insert("yes", ["visible", "goal"])
    right.insert("no", [], " ")

    left = right.yes

    left.insert("yes", [], " ")
    left.insert("no", [], " ")

    # root_node.print()
   # dtree.root.print()
    # return root_node
    return dtree

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    default='MiniGrid-MultiRoom-N6-v0'
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=-1
)
parser.add_argument(
    "--tile_size",
    type=int,
    help="size at which to render tiles",
    default=32
)
parser.add_argument(
    '--agent_view',
    default=False,
    help="draw the agent sees (partially observable view)",
    action='store_true'
)

# -------------- APIs Call for Processing ---------------------

args = parser.parse_args()

env = gym.make(args.env)

dTree = create_tree()

logging.basicConfig(filename='runlog.log')

if args.agent_view:
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

window = Window('gym_minigrid - ' + args.env)
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)

#plot the graph
if len(RUN_STEP) > 0:
    plt.plot(range(0, len(RUN_STEP)), RUN_STEP)
    plt.xlabel("Run #")
    plt.ylabel("Number of Steps Taken")
    plt.savefig('run-vs-step-graph.png')
    plt.show()

    plt.plot(range(0, len(RUN_STEP)), RUN_REWARD)
    plt.xlabel("Run #")
    plt.ylabel("Reward")
    plt.savefig('run-vs-reward-graph.png')
    plt.show()
