import copy
import os
import sys
import urllib.request
from types import MethodType
from typing import Literal

import keras.backend as K
import numpy
import tensorflow as tf
from ChefsHatGym.agents.base_classes.chefs_hat_player import ChefsHatPlayer
from ChefsHatGym.rewards.only_winning import RewardOnlyWinning
from ChefsHatPlayersClub.agents.classic.dql import AgentDQL
from ChefsHatPlayersClub.agents.util.memory_buffer import MemoryBuffer
from keras import Model
from keras.layers import Input, Dense, Lambda, Multiply
from keras.models import load_model
from keras.optimizers import Adam
from keras.src.layers import Add

# Retrieve the actual SumTree class the buffer instantiates
SumTreeClass = type(MemoryBuffer(1, True).buffer)

def _count(self):
    return self.write          # works for both PER and vanilla

SumTreeClass.count = MethodType(_count, SumTreeClass)

# ── Patch MemoryBuffer.sample_batch at runtime ───────────────────────────────
import random, numpy as np
from ChefsHatPlayersClub.agents.util.memory_buffer import MemoryBuffer

def _sample_batch_fixed(self, batch_size):
    batch = []

    # ------- Prioritized Experience Replay branch ---------------------------
    if self.with_per:
        total_p = self.buffer.total()
        if total_p == 0:                     # buffer not initialised yet
            raise ValueError("SumTree total priority is zero – no samples to draw")
        segment = total_p / batch_size       # ← TRUE division fixes the bug

        idx_list = []
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, _, data = self.buffer.get(s)
            batch.append((*data, idx))
            idx_list.append(idx)

        idx = np.array(idx_list, dtype=np.int32)

    # ------- Standard replay branch ----------------------------------------
    elif self.count < batch_size:
        idx = None
        batch = random.sample(self.buffer, self.count)
    else:
        idx = None
        batch = random.sample(self.buffer, batch_size)

    # ------- Reformat for the agent ----------------------------------------
    s_batch            = np.array([e[0] for e in batch])
    a_batch            = np.array([e[1] for e in batch])
    r_batch            = np.array([e[2] for e in batch])
    d_batch            = np.array([e[3] for e in batch])
    new_s_batch        = np.array([e[4] for e in batch])
    possibleActions    = np.array([e[5] for e in batch])
    newPossibleActions = np.array([e[6] for e in batch])

    return (s_batch, a_batch, r_batch, d_batch,
            new_s_batch, possibleActions, newPossibleActions, idx)

# Attach the fixed method
MemoryBuffer.sample_batch = _sample_batch_fixed
# ─────────────────────────────────────────────────────────────────────────────



tf.experimental.numpy.experimental_enable_numpy_behavior()

types = ["Scratch", "vsRandom", "vsEveryone", "vsSelf"]


class AgentRainbow(ChefsHatPlayer):

    # _TYPES: Literal["Scratch", "vsRandom", "vsEveryone", "vsSelf"]

    suffix = "Rainbow_DQL"
    actor = None
    targetNetwork = None
    training = False

    loadFrom = {
        "vsRandom": "/Users/macbook/PycharmProjects/CardPlayingRobot/temp/per/Rainbow_DQL_vsRandom_SelfGeneration500PerDuelingDQNEpisode1/savedModel/actor_PlayerRainbow_DQL_vsRandom_SelfGeneration500PerDuelingDQNEpisode1.keras",
        "vsEveryone": "Trained/dql_vsEveryone.hd5",
        "vsSelf": "Trained/dql_vsSelf.hd5",
    }

    downloadFrom = {
        "vsRandom": "https://github.com/pablovin/ChefsHatPlayersClub/raw/main/src/ChefsHatPlayersClub/agents/classic/Trained/dql_vsRandom.hd5",
        "vsEveryone": "https://github.com/pablovin/ChefsHatPlayersClub/raw/main/src/ChefsHatPlayersClub/agents/classic/Trained/dql_vsEveryone.hd5",
        "vsSelf": "https://github.com/pablovin/ChefsHatPlayersClub/raw/main/src/ChefsHatPlayersClub/agents/classic/Trained/dql_vsSelf.hd5",
    }

    # self, a: int, b: str, c: float, type_: Literal["solar", "view", "both"] = "solar"):

    def __init__(
        self,
        name: str,
        continueTraining: bool = False,
        agentType: Literal["Scratch", "vsRandom", "vsEveryone", "vsSelf"] = "Scratch",
        initialEpsilon: float = 1,
        loadNetwork: str = "",
        saveFolder: str = "",
        verbose_console: bool = False,
        verbose_log: bool = False,
        log_directory: str = "",
    ):
        super().__init__(
            self.suffix,
            agentType + "_" + name,
            this_agent_folder=saveFolder,
            verbose_console=verbose_console,
            verbose_log=verbose_log,
            log_directory=log_directory,
        )

        if continueTraining:
            assert (
                log_directory != ""
            ), "When training an agent, you have to define a log_directory!"

            self.save_model = os.path.join(self.this_log_folder, "savedModel")

            if not os.path.exists(self.save_model):
                os.makedirs(self.save_model)

        self.training = continueTraining
        self.initialEpsilon = initialEpsilon

        self.loadNetwork = loadNetwork

        self.type = agentType
        self.reward = RewardOnlyWinning()

        self.startAgent()

        if not agentType == "Scratch":
            fileName = os.path.join(
                os.path.abspath(sys.modules[AgentDQL.__module__].__file__)[0:-6],
                self.loadFrom[agentType],
            )

            if not os.path.exists(
                os.path.join(
                    os.path.abspath(sys.modules[AgentDQL.__module__].__file__)[0:-6],
                    "Trained",
                )
            ):
                os.mkdir(
                    os.path.join(
                        os.path.abspath(sys.modules[AgentDQL.__module__].__file__)[
                            0:-6
                        ],
                        "Trained",
                    )
                )

            if not os.path.exists(fileName):
                urlToDownload = self.downloadFrom[agentType]
                saveIn = fileName
                print(f"URL Download: {urlToDownload}")
                urllib.request.urlretrieve(self.downloadFrom[agentType], fileName)
            self.loadModel(fileName)

        if not loadNetwork == "":
            self.loadModel(loadNetwork)

    # DQL Functions
    def startAgent(self):
        self.hiddenLayers = 1
        self.hiddenUnits = 256
        self.outputSize = 200
        self.batchSize = 10
        self.tau = 0.52  # target network update rate

        self.gamma = 0.95  # discount rate
        self.loss = "mse"

        self.epsilon_min = 0.1
        self.epsilon_decay = 0.990

        # self.tau = 0.1 #target network update rate

        if self.training:
            self.epsilon = self.initialEpsilon  # exploration rate while training
        else:
            self.epsilon = 0.0  # no exploration while testing

        # behavior parameters
        self.prioritized_experience_replay = True
        self.dueling = True

        QSize = 20000
        self.memory = MemoryBuffer(QSize, self.prioritized_experience_replay)

        # self.learning_rate = 0.01
        self.learning_rate = 0.001

        self.buildModel()

    def buildModel(self):
        self.buildSimpleModel()

        self.actor.compile(
            loss=self.loss,
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=["mse"],
        )

        self.targetNetwork.compile(
            loss=self.loss,
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=["mse"],
        )

    def buildSimpleModel(self):
        """Build Deep Q-Network"""

        def model():

            inputSize = 28
            actionSize = self.outputSize

            state_input = Input(shape=(inputSize,), name="State") # 5 cards in the player's hand + maximum 4 cards in current board
            action_mask = Input(shape=(actionSize,), name="PossibleActions")

            # ── Shared hidden layers ───────────────────────────────────────────────
            x = state_input
            for i in range(self.hiddenLayers + 1):
                x = Dense(self.hiddenUnits * (i + 1),
                          activation="relu",
                          name=f"Dense_shared_{i}")(x)
            if self.dueling:
                # ── Value stream ──────────────────────────────────────────────────
                V = Dense(1,
                          activation="linear",
                          name="Value")(x)  # shape = (batch, 1)

                # ── Advantage stream ─────────────────────────────────────────────
                A = Dense(actionSize,
                          activation="linear",
                          name="Advantage")(x)  # shape = (batch, actionSize)
                A = Lambda(
                    lambda a: a - K.mean(a, axis=1, keepdims=True),
                    output_shape=(actionSize,),
                    name="AdvantageCentered"
                )(A)
                Q = Add(name="Q_values")([V, A])

            else:
                # ── Standard (non-dueling) single head Q ───────────────────────
                Q = Dense(actionSize,
                          activation="linear",
                          name="Q_values")(x)

            Q_masked = Multiply(name="Masked_Q")([action_mask, Q])

            return Model([state_input, action_mask], Q_masked)

        self.actor = model()
        self.targetNetwork = model()

    def loadModel(self, actorModel,targetModel=""):
        """
        Load a saved .keras / .h5 model that contains Lambda layers.
        Sets safe_mode=False to allow deserialization.
        """
        targetModel = "/Users/macbook/PycharmProjects/CardPlayingRobot/temp/per/Rainbow_DQL_vsRandom_SelfGeneration500PerDuelingDQNEpisode1/savedModel/target_PlayerRainbow_DQL_vsRandom_SelfGeneration500PerDuelingDQNEpisode1.keras"
        # 1) load both networks
        self.actor = load_model(actorModel, compile=False, safe_mode=False)
        self.targetNetwork = load_model(targetModel, compile=False, safe_mode=False)

        # 2) re-compile
        self.actor.compile(optimizer="adam", loss="mse")
        self.targetNetwork.compile(optimizer="adam", loss="mse")

    def updateTargetNetwork(self):
        W = self.actor.get_weights()
        tgt_W = self.targetNetwork.get_weights()
        for i in range(len(W)):
            tgt_W[i] = self.tau * W[i] + (1 - self.tau) * tgt_W[i]
        self.targetNetwork.set_weights(tgt_W)

    def updateModel(self, game, thisPlayer):
        """Train Q-network on batch sampled from the buffer"""
        # Sample experience from memory buffer (optionally with PER)
        (
            s,
            a,
            r,
            d,
            new_s,
            possibleActions,
            newPossibleActions,
            idx,
        ) = self.memory.sample_batch(self.batchSize)

        # Apply Bellman Equation on batch samples to train our DDQN
        q = self.actor([s, possibleActions]).numpy()
        next_q = self.actor([new_s, newPossibleActions]).numpy()
        q_targ = self.targetNetwork([new_s, newPossibleActions]).numpy()

        for i in range(s.shape[0]):
            old_q = q[i, a[i]]
            if d[i]:
                q[i, a[i]] = r[i]
            else:
                next_best_action = numpy.argmax(next_q[i, :])
                q[i, a[i]] = r[i] + self.gamma * q_targ[i, next_best_action]

            if self.prioritized_experience_replay:
                # Update PER Sum Tree
                self.memory.update(idx[i], abs(old_q - q[i, a[i]]))

        # Train on batch
        history = self.actor.fit([s, possibleActions], q, verbose=True)

        self.log(
            "-- "
            + self.name
            + ": Epsilon:"
            + str(self.epsilon)
            + " - Loss:"
            + str(history.history["loss"])
        )

    def memorize(
            self,
            state,
            action,
            reward,
            next_state,
            done,
            possibleActions,
            newPossibleActions,
    ):
        """
        Add a transition to the replay buffer.  If using PER, compute the
        initial priority from the TD-error; otherwise use zero.
        """
        if self.prioritized_experience_replay:
            # 1) Prepare 2D inputs for the networks
            state_t = np.expand_dims(np.asarray(state, dtype=np.float32), 0)
            next_t = np.expand_dims(np.asarray(next_state, dtype=np.float32), 0)
            mask_t = np.expand_dims(np.asarray(possibleActions, dtype=np.float32), 0)
            next_mask_t = np.expand_dims(np.asarray(newPossibleActions, dtype=np.float32), 0)

            # 2) Compute Q(s,·) and Q'(s',·)
            q_val = self.actor([state_t, mask_t])
            q_next_targ = self.targetNetwork([next_t, next_mask_t])

            # 3) Double-DQN: pick next action from online net
            best_next = np.argmax(self.actor([next_t, next_mask_t]))

            # 4) Build target and TD-error
            target = reward + self.gamma * (1 - done) * q_next_targ[0, best_next]
            td_error = abs(target - q_val[0, action])

            # Wrap in a 1-element array so memory_buffer.priority(error[0]) works
            per_error = np.array([td_error], dtype=np.float32)
        else:
            # No PER → dummy zero error
            per_error = np.array([0.0], dtype=np.float32)

        # 5) **CRITICAL**: actually write into the buffer!
        #    Signature: (state, action, reward, done, next_state, mask, next_mask, error)
        self.memory.memorize(
            state,  # s
            action,  # a
            reward,  # r
            done,  # d
            next_state,  # s'
            possibleActions,  # mask(s)
            newPossibleActions,  # mask(s')
            per_error  # initial TD-error for PER
        )

    # Agent Chefs Hat Functions

    def get_exhanged_cards(self, cards, amount):
        selectedCards = sorted(cards[-amount:])
        return selectedCards

    def do_special_action(self, info, specialAction):
        return True

    def get_action(self, observations):
        stateVector = numpy.concatenate((observations[0:11], observations[11:28]))
        possibleActions = observations[28:]

        stateVector = numpy.expand_dims(stateVector, 0)
        possibleActions = numpy.array(possibleActions)

        possibleActions2 = copy.copy(possibleActions)

        if numpy.random.rand() <= self.epsilon:
            itemindex = numpy.array(numpy.where(numpy.array(possibleActions2) == 1))[
                0
            ].tolist()
            random.shuffle(itemindex)
            aIndex = itemindex[0]
            a = numpy.zeros(200)
            a[aIndex] = 1

        else:
            possibleActionsVector = numpy.expand_dims(numpy.array(possibleActions2), 0)

            a = self.actor([stateVector, possibleActionsVector])[0]

        return numpy.array(a)

    def get_reward(self, info):
        roles = {"Chef": 0, "Souschef": 1, "Waiter": 2, "Dishwasher": 3}
        print(f"Player names: {info['Player_Names']} ")
        print(f"Player roles: {info['Current_Roles']} ")

        this_player_index = info["Player_Names"].index(self.name)
        this_player_role = info["Current_Roles"][this_player_index]

        try:
            this_player_position = roles[this_player_role]
        except:
            this_player_position = 3

        reward = self.reward.getReward(this_player_position, True)

        # self.log(f"Player names: {this_player_index} - this player name: {self.name}")

        # self.log(
        #     f"this_player: {this_player_index} - Match_Score this player: {info['Match_Score']} - finished players: {info['Finished_Players']}"
        # )
        self.log(f"Finishing position: {this_player_role} - Reward: {reward} ")
        # self.log(
        #     f"REWARD: This player position: {this_player_position} - this player finished: {this_player_finished} - {reward}"
        # )

        return reward

    def update_my_action(self, info):
        if self.training:

            action = info["Action_Index"]
            observation = numpy.array(info["Observation_Before"])
            nextObservation = numpy.array(info["Observation_After"])

            this_player = info["Author_Index"]
            done = info["Finished_Players"][this_player]

            reward = self.get_reward(info)

            state = numpy.concatenate((observation[0:11], observation[11:28]))
            possibleActions = observation[28:]

            next_state = numpy.concatenate(
                (nextObservation[0:11], nextObservation[11:28])
            )
            newPossibleActions = nextObservation[28:]

            self.memorize(
                state,
                action,
                reward,
                next_state,
                done,
                possibleActions,
                newPossibleActions,
            )

    def update_end_match(self, info):
        if not self.training:
            return

        # 1) Log current memory size
        current_mem = self.memory.size()
        self.log(f"-- {self.name}: Memory size before update: {current_mem}")

        # 2) Only train once we have enough samples
        if current_mem > self.batchSize:
            # — Train & update target
            self.updateModel(info["Rounds"], info["Author_Index"])
            self.updateTargetNetwork()

            os.makedirs(self.save_model, exist_ok=True)

            # 3) Save the actor network in Keras format
            actor_path = os.path.join(
                self.save_model,
                f"actor_Player{self.name}.keras"
            )
            try:
                self.actor.save(actor_path)  # .keras → Keras native format
                self.log(f"✅ {self.name}: Saved actor model to {actor_path}")
            except Exception as e:
                self.log(f"❌ {self.name}: Failed to save actor model: {e}")

            # 4) Save the target network in Keras format
            target_path = os.path.join(
                self.save_model,
                f"target_Player{self.name}.keras"
            )
            try:
                self.targetNetwork.save(target_path)
                self.log(f"✅ {self.name}: Saved target model to {target_path}")
            except Exception as e:
                self.log(f"❌ {self.name}: Failed to save target model: {e}")

            # 5) Decay exploration epsilon
            if self.epsilon > self.epsilon_min:
                old_eps = self.epsilon
                self.epsilon *= self.epsilon_decay
                self.log(f"-- {self.name}: Epsilon decayed {old_eps:.4f} → {self.epsilon:.4f}")
        else:
            # Still gathering enough experiences
            self.log(
                f"-- {self.name}: Not enough memory to train "
                f"(have {current_mem}, need > {self.batchSize})"
            )

