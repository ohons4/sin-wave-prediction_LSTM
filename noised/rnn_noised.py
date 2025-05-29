import tensorflow as tf
import numpy as np
import random
from tensorflow.keras import layers, models, optimizers

# バッチを作成
def make_mini_batch(train_data, size_of_mini_batch, length_of_sequences):
  inputs, outputs = [], []
  for _ in range(size_of_mini_batch):
    index   = random.randint(0, len(train_data) - length_of_sequences)
    part    = train_data[index:index + length_of_sequences]
    inputs  = np.append(inputs, part[:, 0])
    outputs = np.append(outputs, part[-1, 1])

  inputs  = np.array(inputs).reshape(-1, length_of_sequences, 1)
  outputs = np.array(outputs).reshape(-1, 1)
  return (inputs, outputs)

# 指定した位置から連続した入力系列を抽出して返す関数。
def make_prediction_initial(train_data, index, length_of_sequences):
  return train_data[index:index + length_of_sequences, 0]

# パラメータ定義
train_data_path              = "/Users/ohons/sin-wave-prediction_LSTM/train_data/noised.npy"
num_of_input_nodes           = 1
num_of_hidden_nodes          = 2
num_of_output_nodes          = 1
length_of_sequences          = 50
num_of_training_epochs       = 2000
length_of_initial_sequences  = 50
num_of_prediction_epochs     = 100
size_of_mini_batch           = 100
learning_rate                = 0.1
forget_bias                  = 1.0
print("train_data_path             = %s" % train_data_path)
print("num_of_input_nodes          = %d" % num_of_input_nodes)
print("num_of_hidden_nodes         = %d" % num_of_hidden_nodes)
print("num_of_output_nodes         = %d" % num_of_output_nodes)
print("length_of_sequences         = %d" % length_of_sequences)
print("num_of_training_epochs      = %d" % num_of_training_epochs)
print("length_of_initial_sequences = %d" % length_of_initial_sequences)
print("num_of_prediction_epochs    = %d" % num_of_prediction_epochs)
print("size_of_mini_batch          = %d" % size_of_mini_batch)
print("learning_rate               = %f" % learning_rate)
print("forget_bias                 = %f" % forget_bias)

# データの読み込み
train_data = np.load(train_data_path)
print("train_data:", train_data)

# 乱数シードの固定
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

# モデルを定義
model = models.Sequential([
  layers.Input(shape=(length_of_sequences, num_of_input_nodes)),
  layers.LSTM(num_of_hidden_nodes),
  layers.Dense(num_of_output_nodes)
])
optimizer = optimizers.SGD(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mse')


# 学習
for epoch in range(num_of_training_epochs):
  inputs, superviors = make_mini_batch(train_data, size_of_mini_batch, length_of_sequences)
  loss = model.train_on_batch(inputs, superviors)

  if (epoch + 1) % 10 == 0:
    print(f"train#{epoch + 1}, train loss: {loss:e}")


# 予測
inputs  = make_prediction_initial(train_data, 0, length_of_sequences)
inputs  = inputs.reshape(1, length_of_sequences, 1)
outputs = []

print("initial:", inputs.flatten())
np.save("initial.npy", inputs.flatten())

for epoch in range(num_of_prediction_epochs):
  output = model.predict(inputs, verbose=0)
  print(f"prediction#{epoch + 1}, output: {output[0, 0]}")
  outputs.append(output[0, 0])

  # シーケンスを1ステップずらして次の入力を作成
  inputs = np.append(inputs[:, 1:, :], output.reshape(1, 1, 1), axis=1)

print("outputs:", outputs)
np.save("outputs.npy", outputs)


# モデルを保存
model.save("data/model")