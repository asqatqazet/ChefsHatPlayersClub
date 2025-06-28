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
from tensorflow.python.keras.layers import Reshape

# Retrieve the actual SumTree class the buffer instantiates
SumTreeClass = type(MemoryBuffer(1, True).buffer)


def _count(self):
    return self.write  # works for both PER and vanilla


SumTreeClass.count = MethodType(_count, SumTreeClass)

# ── Patch MemoryBuffer.sample_batch at runtime ───────────────────────────────
import random, numpy as np
from ChefsHatPlayersClub.agents.util.memory_buffer import MemoryBuffer

def _sample_batch_fixed(self, batch_size):
    batch = []

    if self.with_per:
        total_p = self.buffer.total()
        if total_p == 0:
            raise ValueError("SumTree total priority is zero – no samples to draw")

        segment = total_p / float(batch_size)
        idx_list = []

        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            s = random.uniform(a, b)

            idx, _, data = self.buffer.get(s)

            # NEW ↓↓↓ dereference if needed
            if isinstance(data, (int, np.integer)):
                data = self.buffer.data[data]

            batch.append((*data, idx))
            idx_list.append(idx)

        idx = np.array(idx_list, dtype=np.int32)

    elif self.count < batch_size:
        idx = None
        batch = random.sample(self.buffer, self.count)
    else:
        idx = None
        batch = random.sample(self.buffer, batch_size)

    # unpack to numpy arrays -------------------------------------------------
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

    N_ATOMS = 51

    # Minimum and maximum supports for the return distribution
    V_MIN = -00.0
    V_MAX = 10.0

    # Atom spacing Δz = (V_MAX - V_MIN) / (N_ATOMS - 1)
    delta_z = (V_MAX - V_MIN) / float(N_ATOMS - 1)

    # The fixed tensor of atom locations: shape = (N_ATOMS,)
    z = tf.linspace(V_MIN, V_MAX, N_ATOMS)

    loadFrom = {
        "vsRandom": "/Users/macbook/PycharmProjects/CardPlayingRobot/temp/per/Rainbow_DQL_vsRandom_51PerDuelingDQNEpisode5/savedModel/actor_PlayerRainbow_DQL_vsRandom_51PerDuelingDQNEpisode5.keras",
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

        self.actor.compile(optimizer=Adam(self.learning_rate),
                           loss='categorical_crossentropy')  # ← CE between dists
        self.targetNetwork.compile(optimizer=Adam(self.learning_rate),
                                   loss='categorical_crossentropy')

    def buildSimpleModel(self):
        """Build Deep Q-Network"""

        def model():

            global probs_masked
            inputSize = 28
            actionSize = self.outputSize

            state_input = Input(shape=(inputSize,),
                                name="State")  # 5 cards in the player's hand + maximum 4 cards in current board
            action_mask = Input(shape=(actionSize,), name="PossibleActions")

            # ── Shared hidden layers ───────────────────────────────────────────────
            x = state_input
            for i in range(self.hiddenLayers + 1):
                x = Dense(self.hiddenUnits * (i + 1),
                          activation="relu",
                          name=f"Dense_shared_{i}")(x)
            if self.dueling:
                # ── Value stream ──────────────────────────────────────────────────
                V_logits = Dense(AgentRainbow.N_ATOMS, name="V_logits")(x)  # ⇒ v(s)   (B , N)
                A_logits = Dense(actionSize * AgentRainbow.N_ATOMS, name="A_logits")(x)  # ⇒ concat of a(s,a)
                A_logits = Reshape((actionSize, AgentRainbow.N_ATOMS))(A_logits)  # (B , A , N)

                # ── Advantage stream ─────────────────────────────────────────────

                A_centered = Lambda(lambda a: a - K.mean(a, axis=1, keepdims=True),
                                    name="A_centered")(A_logits)  # eq. (1)

                logits = Add()([tf.expand_dims(V_logits, 1), A_centered])
                # logits shape = (B , A , N)   implements the boxed equation

            else:
                logits = Dense(actionSize * AgentRainbow.N_ATOMS,
                               name="C51_logits")(x)
                logits = Reshape((actionSize, AgentRainbow.N_ATOMS))(logits)

            probs = tf.nn.softmax(logits, axis=-1)  # eq. (2)

            mask3d = Lambda(lambda m: tf.expand_dims(tf.cast(m, tf.float32), -1))(action_mask)
            probs_masked = Multiply()([mask3d, probs])

            return Model([state_input, action_mask], probs_masked)

        self.actor = model()
        self.targetNetwork = model()

    def loadModel(self, actorModel, targetModel=""):
        """
        Load a saved .keras / .h5 model that contains Lambda layers.
        Sets safe_mode=False to allow deserialization.
        """
        targetModel = "/Users/macbook/PycharmProjects/CardPlayingRobot/temp/per/Rainbow_DQL_vsRandom_51PerDuelingDQNEpisode5/savedModel/actor_PlayerRainbow_DQL_vsRandom_51PerDuelingDQNEpisode5.keras"
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

    def updateModel(self, *_):
        (s, a, r, d, ns, mask, nmask, idx) = \
            self.memory.sample_batch(self.batchSize)

        # --- forward passes ----------------------------------------------------
        dist_next = self.targetNetwork([ns, nmask])  # (B , A , N)
        q_next_onl = self._expectation(self.actor([ns, nmask]))
        a_star = tf.argmax(q_next_onl, axis=-1)  # best act by online
        batch_idx = tf.range(self.batchSize, dtype=tf.int64)
        dist_next_a = tf.gather_nd(dist_next,
                                   tf.stack([batch_idx, a_star], axis=1))  # (B , N)

        # --- build target distribution ----------------------------------------
        target_dist = self._project(r, d, dist_next_a)  # (B , N)

        # --- y_true with one-hot over actions -------------------------------
        y_true = tf.one_hot(a, self.outputSize)[:, :, None] * target_dist[:, None, :]

        # --- train step ---------------------------------------------------------
        self.actor.train_on_batch([s, mask], y_true)

        # --- PER priority update ------------------------------------------------
        if self.prioritized_experience_replay:
            pred_dist = self.actor([s, mask])  # (B , A , N)
            pred_a = tf.gather_nd(pred_dist,
                                  tf.stack([batch_idx, a], axis=1))  # (B , N)
            kl = tf.reduce_sum(target_dist *
                               tf.math.log((target_dist + 1e-8) / (pred_a + 1e-8)), axis=-1)
            for i in range(self.batchSize):
                self.memory.update(idx[i], float(kl[i]))

    # ---------------------------------------------------------------------------
    #   DROP-IN REPLACEMENT FOR AgentRainbow.memorize
    # ---------------------------------------------------------------------------
    def memorize(self,
                 state, action, reward, next_state, done,
                 possibleActions, newPossibleActions, error=None):
        """
        Stores a transition in the replay buffer.
        If PER is enabled `error` is the KL-priority; otherwise 0.
        """

        # --------- 1. Compute PER priority (KL divergence) --------------------
        if self.prioritized_experience_replay:
            # -- format inputs as 1-batch tensors
            s_t = np.expand_dims(np.asarray(state, np.float32), 0)
            ns_t = np.expand_dims(np.asarray(next_state, np.float32), 0)
            m_t = np.expand_dims(np.asarray(possibleActions, np.float32), 0)
            nm_t = np.expand_dims(np.asarray(newPossibleActions, np.float32), 0)

            # -- distributions
            dist_pred = self.actor([s_t, m_t])[0]  # pθ(s,·)
            dist_next_o = self.actor([ns_t, nm_t])[0]  # pθ(s',·)
            dist_next_t = self.targetNetwork([ns_t, nm_t])[0]  # pθ⁻(s',·)

            # -- best next action by expected return
            q_next = np.sum(self.z.numpy() * dist_next_o, axis=-1)
            q_next[newPossibleActions == 0] = -1e9
            a_star = int(np.argmax(q_next))

            # -- Φ-projection target distribution
            target_dist = self._project(
                np.asarray([reward], np.float32),
                np.asarray([done], np.float32),
                tf.expand_dims(dist_next_t[a_star], 0)
            )[0].numpy()  # (N,)

            # -- prediction of the action actually taken
            p_sa = dist_pred[action]  # (N,)
            kl = np.sum(target_dist *
                        np.log((target_dist + 1e-8) / (p_sa + 1e-8)))
            per_error = np.array([kl], dtype=np.float32)
        else:
            per_error = 0.0  # vanilla replay

        # --------- 2. Single call to the buffer ------------------------------
        self.memory.memorize(
            state,
            action,
            reward,
            done,
            next_state,
            possibleActions,
            newPossibleActions,
            per_error
        )

    def _project(self, rewards, dones, next_dist):
        # ensure everything’s in float32
        next_dist = tf.cast(next_dist, tf.float32)
        rewards = tf.cast(rewards, tf.float32)
        dones = tf.cast(dones, tf.float32)

        BATCH = tf.shape(rewards)[0]
        z = tf.expand_dims(self.z, 0)  # make sure self.z is a float32 numpy array or tf.constant
        Tz = tf.expand_dims(rewards, 1) \
             + self.gamma * tf.expand_dims(1 - dones, 1) * z
        Tz = tf.clip_by_value(Tz, self.V_MIN, self.V_MAX)

        b = (Tz - self.V_MIN) / self.delta_z
        b = tf.cast(b, tf.float32)  # force float32

        l = tf.cast(tf.floor(b), tf.int32)
        u = tf.cast(tf.math.ceil(b), tf.int32)

        offset = tf.expand_dims(tf.range(BATCH) * self.N_ATOMS, 1)

        m = tf.zeros((BATCH * self.N_ATOMS,), dtype=tf.float32)

        # scatter lower mass
        m = tf.tensor_scatter_nd_add(
            m,
            tf.reshape(l + offset, (-1, 1)),
            tf.reshape(next_dist * (tf.cast(u, tf.float32) - b), (-1,))
        )
        # scatter upper mass
        m = tf.tensor_scatter_nd_add(
            m,
            tf.reshape(u + offset, (-1, 1)),
            tf.reshape(next_dist * (b - tf.cast(l, tf.float32)), (-1,))
        )

        return tf.reshape(m, (BATCH, self.N_ATOMS))

    # Agent Chefs Hat Functions

    def get_exhanged_cards(self, cards, amount):
        selectedCards = sorted(cards[-amount:])
        return selectedCards

    def do_special_action(self, info, specialAction):
        return True

    def _expectation(self, dist):  # dist (B , A , N)
        return tf.reduce_sum(dist * self.z, axis=-1)  # (B , A)

    def get_action(self, observations):
        # ----- 2.1 split observation -------------------------------------------
        state_vec = np.concatenate((observations[0:11], observations[11:28]))  # (28,)
        legal = np.asarray(observations[28:], dtype=np.float32)  # (200,)

        # batchify
        state_vec = state_vec[None, :]  # (1 , 28)
        legal2d = legal[None, :]  # (1 , 200)

        # ----- 2.2 forward pass: distribution ----------------------------------
        dist = self.actor([state_vec, legal2d])[0]  # (A , N)    p_{a,i}

        # ----- 2.3 expectation (eq. 3) -----------------------------------------
        q = np.sum(self.z.numpy() * dist, axis=-1)  # (A,)

        # mask illegal actions by −∞ so they are never chosen greedily
        q[legal == 0] = -1e9

        # ----- 2.4 ε–greedy decision (eq. 7) -----------------------------------
        if np.random.rand() < self.epsilon:
            # choose uniformly from legal indices
            a_idx = int(np.random.choice(np.where(legal == 1)[0]))
        else:
            a_idx = int(np.argmax(q))

        # ----- 2.5 return one-hot vector as before ------------------------------
        one_hot = np.zeros(self.outputSize, dtype=np.float32)
        one_hot[a_idx] = 1.0
        return one_hot

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
