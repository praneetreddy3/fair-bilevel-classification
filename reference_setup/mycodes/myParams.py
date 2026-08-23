
class AlgorithmParams:
    def __init__(self):
        self.dataset_name = 'law' # 'law', 'dutch'  
        self.data_path = get_file_name(self.dataset_name) 
        self.fair_constr = 'SP'  # 'SP', 'EO'
        self.main_verbose_logger = False
        self.constr_imposed_orig = 'Y'
        self.constr_imposed_syn = 'Y'
        self.syn_1_skip = False
        self.syn_1_mainid = [432]
        self.syn_2_skip = True
        self.b_limit_process = False
        # rho: control the penalty term for the constraint
        self.rho_o = 1e3 if self.constr_imposed_orig == 'Y' else 0 #1e6
        self.rho_s = 1e3 if self.constr_imposed_syn == 'Y' else 0   #1e6
        self.rho_o_list = [0, 10, 1e2, 1e3, 1e4]
        self.rho_s_list = [0]

        self.rho_comb_setting = 'allcomb' # 'baseline', 'equal', 'greater', 'allcomb', 'custom'
        self.rho_setting = get_rho_setting(self.rho_comb_setting, self.rho_o_list, self.rho_s_list)
        # lambda: control the penalty term for the regularization
        self.lambda_o = 1e-4 # outer lambda regularization
        self.lambda_i = 1e-4 # inner lambda regularization # 1e-5, 1e-3
        self.lambda_xhat = 0 # 1, 1e-2, 1e-4
        # dataset size and initialization
        self.N_s_same_as_train = True
        self.N_s_ratio = 0.1
        self.N_s = 800
        self.priv_row_ratio, self.upriv_row_ratio = 0.5, 0.5
        self.priv_label_pos, self.priv_label_neg = 0.5, 0.5
        self.upriv_label_pos, self.upriv_label_neg = 0.5, 0.5
        # initialization
        self.init_theta_method = 'user_given' # 'user_given', 'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'normal', 'uniform'
        self.init_xhat = 'normal_tr' # 'normal_tr', 'normal_01', 'all_1'
        self.init_theta = 'all_0' # 'all_0', 'normal_01'
        self.init_xhat_options = ['normal_tr'] # ['normal_tr', 'normal_01', 'all_1']
        self.init_theta_options = ['all_0'] # ['all_0', 'normal_01']
        self.init_shat_yhat = 'as_Orig' # 'as_Setting', 'as_Orig'
        self.init_shat_yhat_options = ['as_Orig'] # 'as_Setting', 'as_Orig'
        # iterative solving setting
        self.alpha_xhat = 0.1 # may be adaptive in the future
        self.is_alpha_xhat_adaptive = True
        self.theta_setting = 'prev_theta'
        self.outer_tol_xhat = 1e-6 # 1e-8, 1e-3
        self.inner_tol = 1e-6
        self.K = 20 # bilevel prob: max iter for updating outer problem (raised from smoke-test default of 3)

    # set user-defined parameters
    def set_params(self, N_s=None, init_xhat=None, init_shat_yhat=None, init_theta=None, 
                   rho_o=None, rho_s=None, max_rho_o=None, max_rho_s=None, 
                   constr_imposed_orig=None, constr_imposed_syn=None):
        if N_s is not None:
            self.N_s = N_s
        if init_xhat is not None:
            self.init_xhat = init_xhat
        if init_shat_yhat is not None:
            self.init_shat_yhat = init_shat_yhat
        if init_theta is not None:
            self.init_theta = init_theta
        if rho_o is not None:
            self.rho_o = rho_o
        if rho_s is not None:
            self.rho_s = rho_s
        if constr_imposed_orig is not None:
            self.constr_imposed_orig = constr_imposed_orig
        if constr_imposed_syn is not None:
            self.constr_imposed_syn = constr_imposed_syn

    def update_terminal_params(self, args):
        if hasattr(args, 'dataset_name') and args.dataset_name is not None:
            self.dataset_name = args.dataset_name
            self.data_path = get_file_name(self.dataset_name)
        if hasattr(args, 'fair_constr') and args.fair_constr is not None:
            self.fair_constr = args.fair_constr
        if hasattr(args, 'rho_o_list') and args.rho_o_list is not None:
            self.rho_o_list = args.rho_o_list
        if hasattr(args, 'rho_s_list') and args.rho_s_list is not None:
            self.rho_s_list = args.rho_s_list    
        if hasattr(args, 'rho_o_list') and args.rho_o_list is not None or \
           hasattr(args, 'rho_s_list') and args.rho_s_list is not None:
            self.rho_setting = get_rho_setting(self.rho_comb_setting, self.rho_o_list, self.rho_s_list)
        if hasattr(args, 'lambda_i') and args.lambda_i is not None:
            self.lambda_i = args.lambda_i
        if hasattr(args, 'lambda_o') and args.lambda_o is not None:
            self.lambda_o = args.lambda_o
        if hasattr(args, 'lambda_xhat') and args.lambda_xhat is not None:
            self.lambda_xhat = args.lambda_xhat
        if hasattr(args, 'N_s') and args.N_s is not None:
            self.N_s = args.N_s
        if hasattr(args, 'syn_1_skip') and args.syn_1_skip is not None:
            self.syn_1_skip = args.syn_1_skip
        if hasattr(args, 'syn_1_mainid') and args.syn_1_mainid is not None:
            self.syn_1_mainid = args.syn_1_mainid
        if hasattr(args, 'syn_2_skip') and args.syn_2_skip is not None:
            self.syn_2_skip = args.syn_2_skip


def get_file_name(dataset_name):
    return "dummy_run"



def get_rho_setting(rho_comb_setting, rho_o, rho_s):
    results = []
    if rho_comb_setting == 'equal':
        results.extend([(ro_o, ro_s) for ro_o in rho_o for ro_s in rho_s if ro_o == ro_s])
    elif rho_comb_setting == 'greater':
        results.extend([(ro_o, ro_s) for ro_o in rho_o for ro_s in rho_s if ro_o > ro_s])
    elif rho_comb_setting == 'allcomb':
        results.extend([(ro_o, ro_s) for ro_o in rho_o for ro_s in rho_s])
    elif rho_comb_setting == 'baseline':
        results.append((0, 0))
    elif rho_comb_setting == 'custom':
        return [(0, 10), (10, 100), (100, 100)]
    return results


class TrainAndPredictParams:
    def __init__(self):
        self.lossFunc = 'LogisticLoss' # 'LogisticLoss'
        self.optimSet = 'L-BFGS' # 'Adam', 'SGD', 'L-BFGS', 'GD'
        self.batch_mode = 'full-batch' # 'full-batch'
        self.batch_size = 1024 # Will be updated if batch_mode = 'full-batch' # 32, 64, 128, 256 
        self.epochs = 20 # L-BFGS: 1, others: >= 1
        self.b_early_stop = False
        self.early_stop_patience = 10
        self.early_stop_threshold = 1e-9
        self.verbose_logger = False
        self.verbose_db = True
        # adaptive stepsize (learning rate)
        self.learning_rate = 1 # 0.01
        self.b_decaying_stepsize = False
        self.decaying_factor = 0.8
        self.decaying_patience = 5
        self.min_lr = 1e-5 # minimum learning rate
        # safety check
        if self.optimSet == 'GD':
            self.batch_mode = 'full-batch'
        if self.optimSet == 'L-BFGS':
            self.epochs = 1

    def set_params(self, learning_rate=None, batch_mode=None, batch_size=None, epochs=None):
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if batch_mode is not None:
            self.batch_mode = batch_mode
            # safety check
            if self.optimSet == 'GD':
                self.batch_mode = 'full-batch'
        if batch_size is not None:
            self.batch_size = batch_size
        if epochs is not None:
            self.epochs = epochs
            # safety check
            if self.optimSet == 'L-BFGS':
                self.epochs = 1


class CTGANParams:
    def __init__(self):
        self.synthesizer_alg = 'dpctgan' # 'ctgan', 'dpctgan'
        self.seed = 42
        self.epochs = 300 # 300
        # (batch_size % pac) has to be 0, otherwise, it will raise an error
        self.batch_size = 500 # 500 
        self.pac = 10 # pac is the parameter for Discriminator
        self.generator_lr = 2e-4 # 2e-4
        self.discriminator_lr = 2e-4 # 2e-4
        self.verbose = False
        self.num_rows = 1000
        self.num_rows_as_real = True
        self.N_s_ratio = 0.1
        self.show_diagnostic = False
        self.show_quality_report = False
        self.plot_column_figs = False # True, False

        # differential privacy
        self.clip_coeff = 0.1      # default = 0.1 # Gradient clipping coefficient - controls the maximum norm of gradients
        self.sigma = 1             # default = 1 # Noise scale - determines the amount of Gaussian noise added for privacy
        self.target_epsilon = 3    # default = 3 # Target privacy budget - upper bound on privacy loss
        self.target_delta = 1e-5   # default = 1e-5 # Target probability of privacy failure - acceptable probability of privacy breach

    def set_params(self, seed=None, synthesizer_alg=None, epochs=None, batch_size=None, pac=None,
                   generator_lr=None, discriminator_lr=None, num_rows=None, clip_coeff=None, sigma=None,
                   target_epsilon=None, target_delta=None):
        if seed is not None:
            self.seed = seed
        if synthesizer_alg is not None:
            self.synthesizer_alg = synthesizer_alg
        if epochs is not None:
            self.epochs = epochs
        if batch_size is not None:
            self.batch_size = batch_size
        if pac is not None:
            self.pac = pac
        if generator_lr is not None:
            self.generator_lr = generator_lr
        if discriminator_lr is not None:
            self.discriminator_lr = discriminator_lr
        if num_rows is not None:
            self.num_rows = num_rows
        if clip_coeff is not None:
            self.clip_coeff = clip_coeff
        if sigma is not None:
            self.sigma = sigma
        if target_epsilon is not None:
            self.target_epsilon = target_epsilon
        if target_delta is not None:
            self.target_delta = target_delta

    def update_terminal_params(self, args):
        if hasattr(args, 'ctgan_epochs') and args.ctgan_epochs is not None:
            self.epochs = args.ctgan_epochs
        if hasattr(args, 'ctgan_batch_size') and args.ctgan_batch_size is not None:
            self.batch_size = args.ctgan_batch_size
        if hasattr(args, 'ctgan_pac') and args.ctgan_pac is not None:
            self.pac = args.ctgan_pac
        if hasattr(args, 'ctgan_G_lr') and args.ctgan_G_lr is not None:
            self.generator_lr = args.ctgan_G_lr
        if hasattr(args, 'ctgan_D_lr') and args.ctgan_D_lr is not None:
            self.discriminator_lr = args.ctgan_D_lr
        if hasattr(args, 'dpctgan_sigma') and args.dpctgan_sigma is not None:
            self.sigma = args.dpctgan_sigma


class FLParams:
    def __init__(self):
        self.fl_method = 'FedAvg' # 'FedAvg', 'FairFed'
        self.fair_metric = None # 'EO', 'SP'
        self.max_iterations = 2   # Server-Client communication rounds            
        self.theta_convergence_tol = 1e-6         
        self.inner_tol = 1e-6
        self.tol_moving_loss_avg = 1e-5
        self.init_theta = 'all_0' # 'all_0', 'normal_01'
        self.num_clients = 1
        self.FairFed_beta = 1 # parameter for FairFed # 5
        self.client_train_data_sizes = None
        self.b_local_RW = False # whether to use local debiasing re-weighting (RW) for clients
        self.local_RWs = None # local debiasing re-weighting (RW) weights

    def set_params(self, fair_metric=None, num_clients=None, client_train_data_sizes=None,
                   local_RWs=None):
        if fair_metric is not None:
            self.fair_metric = fair_metric
        if num_clients is not None:
            self.num_clients = num_clients
        if client_train_data_sizes is not None:
            self.client_train_data_sizes = client_train_data_sizes
        if local_RWs is not None:
            self.local_RWs = local_RWs
        
    def update_terminal_params(self, args):
        if hasattr(args, 'fl_method') and args.fl_method is not None:
            self.fl_method = args.fl_method
        if hasattr(args, 'FairFed_beta') and args.FairFed_beta is not None:
            self.FairFed_beta = args.FairFed_beta
        if hasattr(args, 'b_local_RW') and args.b_local_RW is not None:
            self.b_local_RW = args.b_local_RW



class FLClientParams:
    def __init__(self, fl_method, fair_metric, client_id, round_k, 
                 train_data_size=None, prev_raw_weight=None,
                 cnt_S1Y1=None, cnt_S0Y1=None, 
                 cnt_S1=None, cnt_S0=None,
                 cnt_Yhat1_S0Y1=None, cnt_Yhat1_S1Y1=None,
                 prob_Yhat1_S0Y1=None, prob_Yhat1_S1Y1=None,
                 prob_Yhat1_S0=None, prob_Yhat1_S1=None,
                 acc=None, fair_metric_val=None, 
                 b_local_RW=False, local_RW=None):
        self.fl_method = fl_method
        self.fair_metric = fair_metric
        self.client_id = client_id
        self.round_k = round_k
        self.train_data_size = train_data_size
        self.prev_raw_weight = prev_raw_weight
        self.cnt_S1 = cnt_S1
        self.cnt_S0 = cnt_S0
        self.cnt_S1Y1 = cnt_S1Y1
        self.cnt_S0Y1 = cnt_S0Y1
        self.cnt_Yhat1_S0Y1 = cnt_Yhat1_S0Y1
        self.cnt_Yhat1_S1Y1 = cnt_Yhat1_S1Y1
        self.prob_Yhat1_S0Y1 = prob_Yhat1_S0Y1
        self.prob_Yhat1_S1Y1 = prob_Yhat1_S1Y1
        self.prob_Yhat1_S0 = prob_Yhat1_S0
        self.prob_Yhat1_S1 = prob_Yhat1_S1
        self.acc = acc
        self.fair_metric_val = fair_metric_val
        self.b_local_RW = b_local_RW # default is False
        self.local_RW = local_RW
        

    def set_params(self, fl_method=None, fair_metric=None, client_id=None, round_k=None, 
                   train_data_size=None, prev_raw_weight=None,
                   cnt_S1Y1=None, cnt_S0Y1=None, 
                   cnt_S1=None, cnt_S0=None,
                   cnt_Yhat1_S0Y1=None, cnt_Yhat1_S1Y1=None,
                   prob_Yhat1_S0Y1=None, prob_Yhat1_S1Y1=None,
                   prob_Yhat1_S0=None, prob_Yhat1_S1=None,
                   acc=None, fair_metric_val=None, 
                   b_local_RW=None, local_RW=None):
        if fl_method is not None:
            self.fl_method = fl_method
        if fair_metric is not None:
            self.fair_metric = fair_metric
        if client_id is not None:
            self.client_id = client_id
        if round_k is not None:
            self.round_k = round_k
        if train_data_size is not None:
            self.train_data_size = train_data_size
        if prev_raw_weight is not None:
            self.prev_raw_weight = prev_raw_weight
        if cnt_S1Y1 is not None:
            self.cnt_S1Y1 = cnt_S1Y1
        if cnt_S0Y1 is not None:
            self.cnt_S0Y1 = cnt_S0Y1
        if cnt_S1 is not None:
            self.cnt_S1 = cnt_S1
        if cnt_S0 is not None:
            self.cnt_S0 = cnt_S0
        if cnt_Yhat1_S0Y1 is not None:
            self.cnt_Yhat1_S0Y1 = cnt_Yhat1_S0Y1
        if cnt_Yhat1_S1Y1 is not None:
            self.cnt_Yhat1_S1Y1 = cnt_Yhat1_S1Y1
        if prob_Yhat1_S0Y1 is not None:
            self.prob_Yhat1_S0Y1 = prob_Yhat1_S0Y1
        if prob_Yhat1_S1Y1 is not None:
            self.prob_Yhat1_S1Y1 = prob_Yhat1_S1Y1
        if prob_Yhat1_S0 is not None:
            self.prob_Yhat1_S0 = prob_Yhat1_S0
        if prob_Yhat1_S1 is not None:
            self.prob_Yhat1_S1 = prob_Yhat1_S1
        if acc is not None:
            self.acc = acc
        if fair_metric_val is not None:
            self.fair_metric_val = fair_metric_val
        if b_local_RW is not None:
            self.b_local_RW = b_local_RW
        if local_RW is not None:
            self.local_RW = local_RW
        