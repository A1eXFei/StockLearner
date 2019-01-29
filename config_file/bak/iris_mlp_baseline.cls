[Model]
name = iris_mpl_baseline
type = MLP

[Input]
unit = 4

[Layer1]
unit = 10
act_fn = tf.nn.relu6

[Layer2]
unit = 10
keep_prob = 0.9
act_fn = tf.nn.relu6

[Output]
unit = 3

[Dataset]
batch_size = 200
repeat_time = -1

[Hyper Parameters]
type = classification
learning_rate = 0.0001
decay_learning_rate = false
decay_rate = 0.9
decay_step = 100
log_dir = mlp/
loss_fn = cross_entropy
l2_loss = false
opt_fn = tf.train.AdamOptimizer
acc_fn = correct_prediction
model_dir = mlp/
echo = 10000
enable_tensorboard_log = true