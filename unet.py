


layers_dict = {}
number_of_layers = 3
number_of_final_layers = 2
number_of_start_conv = 2

dropout_rate = 0.0
#reg_rate = 0.00005


input_net = Input((64, 64, 3))

n0 = Normalization()(input_net)


layers_dict["Cnvl0"] = Conv2D(32, (3, 3), activation = "softplus", padding = "same")(n0)


for i in range(number_of_start_conv):
    #print(i)
    layers_dict["Cnvl" + str(i + 1)] = Conv2D(32, (3, 3), activation = "softplus", padding = "same")(layers_dict["Cnvl" + str(i)])
    layers_dict["Cnvl" + str(i + 1)] = Dropout(dropout_rate)(layers_dict["Cnvl" + str(i + 1)])


layers_dict["cnvl0"] = Conv2D(32, (3, 3), activation = "softplus", padding = "same")(layers_dict["Cnvl" + str(i + 1)])
layers_dict["dp0"] = MaxPool2D((2, 2), (2, 2), padding = "valid")(layers_dict["cnvl0"])



for i in range(1, number_of_layers):
    layers_dict["cnvl" + str(i)] = Conv2D(int(32 * 2**i), (3, 3), activation = "softplus", padding = "same")(layers_dict["dp" + str(i-1)])
    layers_dict["mp" + str(i)] = MaxPool2D((2, 2), (2, 2), padding = "valid")(layers_dict["cnvl" + str(i)])
    layers_dict["dp" + str(i)] = Dropout(dropout_rate)(layers_dict["mp" + str(i)])
    

layers_dict["dpT" + str(i)] = Conv2DTranspose(int(32 * 2**i), (2, 2), strides = (2, 2), activation = "softplus", padding = "same")(layers_dict["dp" + str(i)])
    

for i in reversed(range(0, number_of_layers -1)):
    layers_dict["cnvl_T" + str(i)] = Conv2DTranspose(int(32 * 2**i), (2, 2), strides = (2, 2), activation = "softplus", padding = "same")(layers_dict["dpT" + str(i+1)])
    layers_dict["CT" + str(i)] = concatenate([layers_dict["cnvl_T" + str(i)], layers_dict["cnvl" + str(i)]], axis = 3)
    layers_dict["dpT" + str(i)] = Dropout(dropout_rate)(layers_dict["cnvl_T" + str(i)])
    
layers_dict["dpF0"] = Conv2D(32, (3, 3), activation = "softplus", padding = "same")(layers_dict["cnvl_T" + str(i)])


for i in range(1, number_of_final_layers):
    layers_dict["cnvlF" + str(i)] = Conv2D(max(32 / 2**i, 3), (3, 3), activation = "softplus", padding = "same")(layers_dict["dpF" + str(i -1)])
    layers_dict["dpF" + str(i)] = Dropout(0.2)(layers_dict["cnvlF" + str(i)])
    
i += 1
layers_dict["cnvlF" + str(i)] = Conv2D(1, (3, 3), activation = "softplus", padding = "same")(layers_dict["cnvlF" + str(i -1)])
i+= 1
#layers_dict["cnvlF" + str(i)] = tf.concat((layers_dict["cnvlF" + str(i-1)], input_net), axis = 3)
#i += 1

layers_dict["cnvlF" + str(i)] = Conv2D(1, (3, 3), activation = "sigmoid", padding = "same")(layers_dict["cnvlF" + str(i -1)])


model = Model(inputs = input_net, outputs = layers_dict["cnvlF" + str(i)])
    
    
    
    
