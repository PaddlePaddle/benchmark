import parl.layers as layers
from paddle import fluid
import threading as th

class DDPGAgent(object):
    def __init__(self, algorithm, no_mem_allocation=False):
        self.alg = algorithm
        self.obs_dim = self.alg.obs_dim
        self.act_dim = self.alg.act_dim
        self.ensemble_num = self.alg.ensemble_num
        
        self._no_mem_allocation = no_mem_allocation
        self._define_program()
        self.place = fluid.CPUPlace() if self.alg.gpu_id < 0 \
                else fluid.CUDAPlace(self.alg.gpu_id)
        self.fluid_executor = fluid.Executor(self.place)
        self.fluid_executor.run(fluid.default_startup_program())
        
        use_cuda = True if self.alg.gpu_id >= 0 \
                else False
        self.parallel_executors = []
        self.scopes = []
        for i in range(self.ensemble_num):
            new_scope = fluid.global_scope().new_scope()
            with fluid.scope_guard(new_scope):
                exec_strategy = fluid.ExecutionStrategy()
                exec_strategy.use_experimental_executor = True
                exec_strategy.num_threads = 4

                build_strategy = fluid.BuildStrategy()
                build_strategy.remove_unnecessary_lock = True
                #build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce
                trainer = fluid.ParallelExecutor(
                        use_cuda=use_cuda, 
                        loss_name=self.learn_programs_output[i],
                        main_program=self.learn_programs[i],
                        exec_strategy=exec_strategy,
                        build_strategy=build_strategy)
                self.parallel_executors.append(trainer)
                self.scopes.append(new_scope)


    def _define_program(self):
        self.actor_predict_programs = []
        self.actor_predict_outputs = []
        self.learn_programs = []
        self.learn_programs_output = []
        for i in range(self.ensemble_num):
            actor_predict_program = fluid.Program()
            with fluid.program_guard(actor_predict_program):
                obs = layers.data(name='obs', shape=[self.obs_dim], dtype='float32')
                action = self.alg.actor_predict(obs, model_id=i)
            self.actor_predict_programs.append(actor_predict_program)
            self.actor_predict_outputs.append([action])

            learn_program = fluid.Program()
            with fluid.program_guard(learn_program):
                obs = layers.data(name='obs', shape=[self.obs_dim], dtype='float32')
                action = layers.data(name='action', shape=[self.act_dim], dtype='float32')
                reward = layers.data(name='reward', shape=[], dtype='float32')
                next_obs = layers.data(name='next_obs', shape=[self.obs_dim], dtype='float32')
                terminal = layers.data(name='terminal', shape=[], dtype='bool')
                actor_lr = layers.data(name='actor_lr', shape=[1], dtype='float32', append_batch_size=False)
                critic_lr = layers.data(name='critic_lr', shape=[1], dtype='float32', append_batch_size=False)
                critic_loss = self.alg.learn(obs, action, reward, next_obs, terminal, actor_lr, critic_lr, model_id=i)
            if self._no_mem_allocation:
                for var in learn_program.blocks[0].vars:
                    if not learn_program.blocks[0].var(var).is_data: 
                        learn_program.blocks[0].var(var).persistable = True
            self.learn_programs.append(learn_program)
            self.learn_programs_output.append([critic_loss.name])

    def actor_predict(self, obs, model_id):
        # Make multi-thread safe
        thread_name = '_predict_{}'.format(model_id)

        feed = {'obs': obs}
        action = self.fluid_executor.run(self.actor_predict_programs[model_id],
                                         feed=feed,
                                         fetch_list=self.actor_predict_outputs[model_id],
                                         feed_var_name='feed' + thread_name,
                                         fetch_var_name='fetch' + thread_name,
                                         )[0]
        return action

    def learn(self, obs, action, reward, next_obs, terminal, actor_lr, critic_lr, model_id, need_fetch=False):
        feed = {'obs': obs,
                'action': action,
                'reward': reward,
                'next_obs': next_obs,
                'terminal': terminal,
                'actor_lr': actor_lr,
                'critic_lr': critic_lr}
            
        feed = [feed]
        if need_fetch: 
            critic_loss = self.parallel_executors[model_id].run(
                            feed=feed,
                            fetch_list=self.learn_programs_output[model_id])[0]
        else:
            self.parallel_executors[model_id].run(feed=feed, fetch_list=[])
            critic_loss = None

        return critic_loss
