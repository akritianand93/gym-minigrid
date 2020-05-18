import random
import logging
from collections import deque

DISCOUNT_FACTOR = 0.9     # D0513
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000
BATCH_SIZE = 20

ROOT_FLAG = False

ACTION_TO_IDX = {
    0 : 'left',
    1 : 'right',
    2 : 'forward',
    3 : 'pickup',
    4 : 'drop',
    5 : 'toggle',
    6 : 'enter'
}

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
            print("Test node ", self.predicate, self.obj)
        else:
            print("")
            print("Leaf Node ", self.assertAction, self.expression, self.Q_val)
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

    def insert_new(self, path, val, assertAction=" "):
        # Compare the new value with the parent node
        node = self
        if len(path) > 1:
            for i in range(len(path)-1):
                if path[i] == "yes":
                    node = node.yes
                else:
                    node = node.no
                # node = node.p

        # print(node.predicate, node.obj)
        if len(val) != 0:
            # print(path[-1])
            if path[-1] == "yes":
                node.yes = TreeNode(val[0], val[1])
                node.yes.parent = node
            else:
                node.no = TreeNode(val[0], val[1])
                node.no.parent = node
        else:
            # print(path[-1])
            if path[-1] == "yes":
                node.yes = TreeNode("", "", 1, assertAction)
                node.yes.parent = node
            else:
                node.no = TreeNode("", "", 1, assertAction)
                node.no.parent = node


def create_tree():
    dtree = Tree()
    # 1
    dtree.root = TreeNode("visible", "key")
    # root_node = TreeNode("visible", "key")

    # 2 ad 3
    dtree.root.insert_new(["yes"], ["carrying", "key"])
    dtree.root.insert_new(["no"], ["visible", "door"])

    # 2 -> 4 and 5
    dtree.root.insert_new(["yes", "yes"], ["visible", "door"])
    dtree.root.insert_new(["yes", "no"], ["locked", "door"])

    # 3 -> 6 and 7
    dtree.root.insert_new(["no", "yes"], ["locked", "door"])
    dtree.root.insert_new(["no", "no"], ["visible", "goal"])

    # 4 -> 8 and 9
    dtree.root.insert_new(["yes", "yes", "yes"], ["locked", "door"])
    dtree.root.insert_new(["yes", "yes", "no"], ["visible", "goal"])

    # 5 -> 10 and 11
    dtree.root.insert_new(["yes", "no", "yes"], [], " ")
    dtree.root.insert_new(["yes", "no", "no"], ["visible", "goal"])

    # 6 -> 12 and 13
    dtree.root.insert_new(["no", "yes", "yes"], ["carrying", "key"])
    dtree.root.insert_new(["no", "yes", "no"], ["visible", "goal"])

    # 7 -> 14 and 15
    dtree.root.insert_new(["no", "no", "yes"], [], " ")
    dtree.root.insert_new(["no", "no", "no"], [], " ")

    # 8 -> 16 and 17
    dtree.root.insert_new(["yes", "yes", "yes", "yes"], [], " ")
    dtree.root.insert_new(["yes", "yes", "yes", "no"], ["visible", "goal"])

    # 9 -> 18 and 19
    dtree.root.insert_new(["yes", "yes", "no", "yes"], [], " ")
    dtree.root.insert_new(["yes", "yes", "no", "no"], [], " ")

    # 10 -> leaf node
    # 11 -> - and -
    dtree.root.insert_new(["yes", "no", "no", "yes"], [], " ")
    dtree.root.insert_new(["yes", "no", "no", "no"], [], " ")

    # 12 -> 20 and 21
    dtree.root.insert_new(["no", "yes", "yes", "yes"], [], " ")
    dtree.root.insert_new(["no", "yes", "yes", "no"], [], " ")

    # 13 -> 22 and 23
    dtree.root.insert_new(["no", "yes", "no", "yes"], [], " ")
    dtree.root.insert_new(["no", "yes", "no", "no"], [], " ")

    # 14 -> leafnode
    # 15 -> leafnode
    # 16 -> leafnode

    # 17 -> 24 and 25
    dtree.root.insert_new(["yes", "yes", "yes", "no", "yes"], [], " ")
    dtree.root.insert_new(["yes", "yes", "yes", "no", "no"], [], " ")

    # 18 -> leafnode
    # 19 -> leafnode
    # 20 -> leafnode
    # 21 -> leafnode
    # 22 -> leafnode
    # 23 -> leafnode
    # 24 -> leafnode
    # 25 -> leafnode

    return dtree