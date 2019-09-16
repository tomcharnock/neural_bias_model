
# Neural physical engines for inferring the halo mass distribution function
## Tom Charnock

Here is an outlined sketch for the neural physical engine and mixture density network used as the neural bias model. The code is written in `julia` and `python`. For convenience we are using SoS notebooks here to be able to use both languages.

The idea is to build a neural network respecting rotational symmetries and whoes output is a mixture density network (of Gaussians). This will receive a LPT density field and produce the halo mass distribution function.

Note that we are using TensorFlow 1.13.1 in both `python` and `julia`.

We will first load the `multipole_kernels` module in python purely to get the indices for filling the rotationally invariant weight kernel. We will consider a single $\ell=0$ kernel (completely rotationally invariant) with extent $3\times3\times3$. The module is available at https://github.com/tomcharnock/multipole_kernels.

We are going to get the number of parameters, and number of biases in the kernels for book-keeping and then pull out the `kernel_indices`, `weight_index` and `kernel_shape`. The `kernel_indices` contains an `[N, d+2]` integer array which describes the index of every element of the `d`-dimensional set of kernels (much like meshgrid). The `weight_index` is an `[N]` integer array whose values correspond to the index of the parameter which will be populated into the kernel. Finally `kernel_shape` is simply the shape of the kernel.


```julia
%use python
import tensorflow as tf
from multipole_kernels.multipole_kernels import multipole_kernels as mk
MK = mk(kernel_size=[3, 3, 3], ℓ=[0], input_filters=1)

num_kernel_params = MK.num_params
num_bias_params = MK.num_output_filters

sess = tf.Session()
sess.run(tf.global_variables_initializer())

kernel_indices = sess.run("indices:0")
weight_index = sess.run("weight_index:0")
kernel_shape = sess.run("shape:0")
```

We now load the necessary modules in `julia`


```julia
%use julia
using TensorFlow; tf = TensorFlow
using SpecialFunctions
using NPZ
using Distributions
```

and we load the `python` variable into the environment. Note that we add 1 to the indices variables since `julia` is 1-indexed.


```julia
%get num_kernel_params, num_bias_params, kernel_indices, weight_index, kernel_shape
kernel_indices .+= 1;
weight_index .+= 1;
```

We want to now define the mixture density network (MDN). We can represent the fully connected layers as a list of the number of nodes, for example `layers = [10, 16, 16, 1]` would take an input of shape `10` and have two hidden layers of shape `16` and output a single entry vector. In our example we will have no fully connected layers between the convolutional kernel and MDN. We can write this as


```julia
distributions = 2;
layers = [1];
```

The total number of parameters for the MDN can then be calculated using the `get_mdn_params(layers, distributions)` function which loops through the layers and counts the number of parameters needed for the fully connected network.


```julia
function get_mdn_params(layers, distributions)
    number_of_parameters = 0
    for i = 1: length(layers) - 1
        number_of_parameters += prod(layers[i: i + 1]) + layers[i + 1]
    end
    return number_of_parameters, layers[end] * distributions
end
```




    get_mdn_params (generic function with 1 method)



The total number of parameters in the network is then given by the number of fully connected parameters, the number of inputs to the distributions and the number of kernel parameters and bias parameters from the `multipole_kernels`. Since we are using Gaussians in this example we have three parameters per distribution per element of the output vector of the fully connected layers, one amplitude, one mean and one standard deviation.


```julia
fc_params, mdn_kernels = get_mdn_params(layers, distributions)
num_params = fc_params + (mdn_kernels + distributions) * 3 + num_kernel_params + num_bias_params
```




    17



We can now define our parameter variable in TensorFlow. For convenience we will initialise the variable with random values from a normal distribution.


```julia
x = Normal(0., 1.)
p = tf.Variable(
    Float64.(rand(x, num_params)),
    trainable=true,
    name="parameters")
```




    Variable{Float64}(<Tensor parameters:1 shape=(17) dtype=Float64>, <Tensor parameters/Assign:1 shape=(17) dtype=Float64>)



So that we can freely assign to the parameters variable we also define these operations


```julia
update_p = tf.placeholder(
    Float64, 
    shape=Int64[num_params],
    name="update_parameters")
assign_p = tf.assign(
    tf.get_tensor_by_name("parameters"),
    update_p,
    name="assign_parameters")
```




    <Tensor assign_parameters:1 shape=(17) dtype=Float64>



We also can define a coordinate transformation in the network parameters such that their central values are approximately zero. This is useful for defining a simple prior later. We will choose the initial amplitude to be `initial_amplitude=log(1e-3)`, the mass threshhold to be `log_mass_threshhold=log(2e12)` and the initial width to be `initial_width=log(1e3)`.


```julia
initial_amplitude = tf.Variable(
    Float64(log(1e-3)),
    trainable=false,
    name="initial_amplitude")
logMth = tf.Variable(
    Float64(log(2e12)),
    trainable=false, 
    name="logMth")
initial_width = tf.Variable(
    Float64(log(1e3)),
    trainable=false, 
    name="initial_width")
```




    Variable{Float64}(<Tensor initial_width:1 shape=() dtype=Float64>, <Tensor initial_width/Assign:1 shape=() dtype=Float64>)



We also need to include the volume of a pixel on the grid. In the paper we consider a 250$h^{-1}$Mpc patch gridded onto a $64^3$ grid.


```julia
L0 = Float64(250.)
L1 = Float64(250.)
L2 = Float64(250.)
N0 = Float64(64.)
N1 = Float64(64.)
N2 = Float64(64.)
V = tf.Variable(
    (L0 / N0) * (L1 / N1) * (L2 / N2),
    trainable=false,
    name="V")  
```




    Variable{Float64}(<Tensor V:1 shape=() dtype=Float64>, <Tensor V/Assign:1 shape=() dtype=Float64>)



Finally, we include the width of the bias prior. We are going choose a Gaussian prior centred on zero for all of the parameters and so the width needs to be large enough to allow for freedom for the parameters to explore, but tight enough that the momenta of the parameters doesn't get too large in the HMC sampling. We use `bias_width=10`, we we found not to cause problems, although we did not study this in detail. All parameters were well within this prior at the end of the inference.


```julia
bias_prior = tf.Variable(
    Float64(10.), 
    trainable=false, 
    name="bias_prior")
```




    Variable{Float64}(<Tensor bias_prior:1 shape=() dtype=Float64>, <Tensor bias_prior/Assign:1 shape=() dtype=Float64>)



Now we can set up the graph. We will be offloading the work normally done by the `multipole_kernels` module in `python` to the TensorFlow frontend for `julia`. This will allow us to show the details for how the kernels are filled. We start by making the `kernel_indices`, `weight_index` and `shape` pulled in from the `python` module into TensorFlow variables. Remember the values of `kernel_indices` and `weight_index` need to be increased by one since `julia` is 1-indexed.


```julia
kernel_indices = tf.Variable(
    Int64.(kernel_indices),
    trainable=false,
    name="kernel_indices")
weight_index = tf.Variable(
    Int64.(weight_index),
    trainable=false,
    name="weight_index")
shape = tf.Variable(
    Int64.(kernel_shape),
    trainable=false,
    name="shape")
```




    Variable{Int64}(<Tensor shape:1 shape=(5) dtype=Int64>, <Tensor shape/Assign:1 shape=(5) dtype=Int64>)



First we take a slice out of the parameters variable (of size `num_kernel_params`) for the weights of the kernel and place these values in a tensor according to the index values in `weight_index` which describes where each parameter will be populated into the final `multipole_kernel`. For book-keeping we will introduce a counting variable (`start`) for how many of the parameters have been used.


```julia
start = 1
full_weights = tf.gather(
        tf.slice(
            p,
            [start],
            [num_kernel_params]),
        weight_index)
start += num_kernel_params
```




    5



And now we populate the kernel by placing the values into the convolutional kernel using the indices described in `kernel_indices`. Unfortunately, the `julia` frontend to TensorFlow doesn't recognise the shape correctly - I haven't implemented the shape inference yet. As a short work around if you want to see the shapes of the tensors in this example I have made a fake tensor which won't work but has the correct shape, just set `see_shape=true`. 


```julia
see_shape=false
if see_shape
    kernel = tf.placeholder(
        Float64, 
        shape=Int64[3, 3, 3, 1, 1],
        name="kernel")
else
    kernel = tf.scatter_nd(
        kernel_indices, 
        full_weights, 
        shape,
        name="kernel")
end
```




    <Tensor kernel:1 shape=(3, 3, 3, 1, 1) dtype=Float64>



We also need to get the parameters for the bias of the convolution


```julia
biases = tf.identity(
    tf.slice(
        p,
        [start],
        [num_bias_params]),
    name="biases")
start += num_bias_params
```




    6



The parameters for the fully connected network (which in our case is zero parameters)


```julia
fully_connected = tf.identity(
    tf.slice(
        p,
        [start],
        [fc_params]),
    name="fully_connected")
start += fc_params
```




    6



And now the parameters for the amplitude, mean and standard deviations of both of the distributions of the mixture of Gaussians


```julia
amplitude_kernels = tf.reshape(
    tf.slice(p,
        [start],
        [mdn_kernels]),
    Int64[layers[end], distributions],
    name="amplitude_kernels")
start += mdn_kernels

amplitude_biases = tf.identity(
    tf.slice(
        p,
        [start],
        [distributions]),
    name="amplitude_biases")
start += distributions

mean_kernels = tf.reshape(
    tf.slice(p,
        [start],
        [mdn_kernels]),
    Int64[layers[end], distributions],
    name="mean_kernels")
start += mdn_kernels

mean_biases = tf.identity(
    tf.slice(
        p,
        [start],
        [distributions]),
    name="mean_biases")
start += distributions

std_kernels = tf.reshape(
    tf.slice(
        p,
        [start],
        [mdn_kernels]),
    Int64[layers[end], distributions],
    name="std_kernels")
start += mdn_kernels
std_biases = tf.identity(
    tf.slice(
        p,
        [start],
        [distributions]),
    name="std_biases") 
```




    <Tensor std_biases:1 shape=(2) dtype=Float64>



Now we want to actually construct the network. Firstly the convolution is simple in this case (note that this function will need to be modified if several layers are used). We take the logarithm of the overdensity field as the input to the neural bias model and then perform the convolution and use a softplus activation function.


```julia
function convolution(x)
    kernel = tf.get_tensor_by_name("kernel")
    b = tf.get_tensor_by_name("biases")
    x = tf.log(tf.add(Float64(1.0), x))
    x = tf.add(
            tf.nn.conv3d(
                x, 
                kernel, 
                strides=[1, 1, 1, 1, 1], 
                padding="VALID"), 
            b)
    return tf.reduce_sum(tf.nn.softplus(x), axis=5)
end
```




    convolution (generic function with 1 method)



Then we have the MDN (which also builds a fully connected network if there is one). We use softplus activation here too.


```julia
function build_fully_connected(x, layers, distributions)
    fully_connected = tf.get_tensor_by_name("fully_connected")
    c1 = Int64(0)
    size = Int64(0)
    for i=1:length(layers)-1
        c1 = c1 + size
        size = prod(layers[i] * layers[i + 1])
        w = tf.reshape(
                tf.slice(fully_connected, 
                         [c1+1], 
                         [size]), 
                Int64[layers[i], layers[i+1]])
        x = tf.matmul(x, w)
        c1 = c1 + size
        size = layers[i+1]
        b = tf.slice(fully_connected, 
                             [c1+1], 
                             [size])
        x = tf.add(x, b)
        x = tf.nn.softplus(x)
    end
    return x
end
```




    build_fully_connected (generic function with 1 method)



Constructing the MDN can then simply be don by calling `build_network`. This separates the output of the fully connected network into the amplitudes, means and standard deviations. Because, unlike with a normal MDN, we are not using the softmax function on the amplitudes, we instead break the degeneracy by ordering the means by size. To do so, we unpack the tensor, and add each subsequent mean to the previous, cutoff using a relu function and then restacked. Both the amplitude and the standard deviation are kept positive using the softplus function.


```julia
function build_network(x, layers, distributions)
    x = build_fully_connected(x, layers, distributions)

    xα = tf.matmul(x, tf.get_tensor_by_name("amplitude_kernels"))
    bα = tf.add(
        tf.get_tensor_by_name("amplitude_biases"),
        tf.get_tensor_by_name("initial_amplitude"))
    xα = tf.add(xα, bα)
    α = tf.nn.softplus(xα)

    xμ = tf.matmul(x, tf.get_tensor_by_name("mean_kernels"))
    μs = tf.split(
        2, 
        distributions, 
        tf.add(
            xμ, 
            tf.get_tensor_by_name("mean_biases")))
    μ = Array{Any}(undef, distributions)
    μ[1] = tf.dropdims(
        tf.add(
            μs[1], 
            tf.get_tensor_by_name("logMth")), 
        dims=Int64[2])
    for i = 1: distributions - 1
        μ[i+1] = tf.add(
            μ[i], 
            tf.dropdims(
                tf.nn.relu(μs[i+1]), 
                dims=Int64[2]))
    end
    μ = tf.stack(μ, axis=2)

    xσ = tf.matmul(
        x,
        tf.get_tensor_by_name("std_kernels"))
    bσ = tf.add(
        tf.get_tensor_by_name("std_biases"),
        tf.get_tensor_by_name("initial_width"))
    xσ = tf.add(xσ, bσ)
    σ = tf.nn.softplus(xσ)
    return α, μ, σ
end
```




    build_network (generic function with 1 method)



These parameters from the MDN are inserted into the Gaussian function, which gives us a tensor with 2 Gaussians evaluated at a given mass and density environment (via the parameters of the MDN). We will at this point define two functions for subtract and divide since they are not defined in the `julia` TensorFlow frontend. Note that this is not necessary, but it does make for better consistency.


```julia
function subtract(x, y)
    return x .- y
end

function divide(x, y)
    return x ./ y
end
```




    divide (generic function with 1 method)



Note the normalisation with respect to the mean.


```julia
function gaussian(α, μ, σ, u)
    return tf.multiply(
        divide(
            α, 
            tf.exp(μ)), 
        divide(
            tf.exp(
                tf.multiply(
                    Float64(-0.5),
                    tf.square(
                        divide(
                            subtract(
                                tf.expand_dims(u, 2), 
                                μ),
                            σ)))), 
            tf.sqrt(
                tf.multiply(
                    Float64(2. * pi),
                    tf.square(σ)))))
end
```




    gaussian (generic function with 1 method)



We can now finally input our input field into the graph. First, the density field evolved using LPT is passed as part of the `BORG` algorithm. We therefore want to define a placeholder which can be filled on each evaluation of the likelihood. It's shape is that of the grid


```julia
δ_LPT = tf.placeholder(
    Float64, 
    shape=Int64[N0, N1, N2], 
    name="density_LPT")
```




    <Tensor density_LPT:1 shape=(64, 64, 64) dtype=Float64>



Next we convolve this using the `multipole_kernels`. To do so we need to add an extra dimension to the beginning and end of the density field tensor (beginning acts like a batch size and end acts as an input channel size). The output of the convolutional kernel needs to be reshaped into a flat tensor of predictions for each central voxel of the receptive input patch.


```julia
δ_NPE = TensorFlow.reshape(
    convolution(
        TensorFlow.expand_dims(
            TensorFlow.expand_dims(δ_LPT, 1), 
            5)), 
    Int64[(N0 - 2) * (N1 - 2) * (N2 - 2), 1],
    name="density_NPE")
```




    <Tensor density_NPE:1 shape=(238328, 1) dtype=Float64>



This is then passed through the MDN and we collect the parameters as named tensors for use in `BORG`.


```julia
δ_α, δ_μ, δ_σ = build_network(
    δ_NPE,
    layers, 
    distributions)
δ_α = tf.identity(δ_α, name="density_amplitude")
δ_μ = tf.identity(δ_μ, name="density_mean")
δ_σ = tf.identity(δ_σ, name="density_covariance")
```




    <Tensor density_covariance:1 shape=(238328, 2) dtype=Float64>



We do not have to evaluate the Gaussian with these outputs since they are integrated from some mass threshhold, and this can be done analytically. Details of the form of the function is written in the paper.


```julia
error_function = tf.identity(
    erfc(
        divide(
            subtract(
                tf.get_tensor_by_name("logMth"), 
                tf.add(δ_μ, tf.square(δ_σ))), 
            tf.sqrt(tf.multiply(Float64(2.), tf.square(δ_σ))))), 
        name="error_function")
normalisation_factor = tf.multiply(
    tf.multiply(Float64(0.5), δ_α),
    tf.exp(tf.multiply(Float64(0.5), tf.square(δ_σ))),
    name="factor")
integral = tf.reduce_sum(
    tf.multiply(normalisation_factor,
                error_function),
    name="integral")
```




    <Tensor integral:1 shape=() dtype=Float64>



The data is in the form of a halo catalogue (or several catalogues), with selection masks. The masses of the halos should be the logarithm in an array where the first dimension describes each catalogue. The mass is a boolean of the same form, where true will not be masked and false will be masked. Because of the form of these tensors, the `max_catalogue_length` is the length of the longest catalogue. For simplicity we will define a single catalogue with `max_catalogue_length=3000`.


```julia
max_catalogue_length = 3000
num_catalogues = 1
log_mass = Float64.(ones((num_catalogues, max_catalogue_length)))
data_mask = Bool.(ones((num_catalogues, max_catalogue_length)));
```

The corresponding tensors are therefore


```julia
catalogue_u = tf.Variable(
    log_mass,
    trainable=false,
    name="log_mass")
mask = tf.Variable(
    data_mask,
    trainable=false,
    name="mask")
```




    Variable{Bool}(<Tensor mask:1 shape=(1, 3000) dtype=Bool>, <Tensor mask/Assign:1 shape=(1, 3000) dtype=Bool>)



The indices of the halos is a bit tricky to process. Since we want to take the 3x3x3 local patch around the central voxel we need to add the extra indices into the tensor. This can be done for each catalogue using (note that I have only written this for 3x3x3 at the moment, but it is very easy to extend to arbitrary sizes).


```julia
function process_indices(indices)
    all_indices = Array{Int64, 2}(undef, 27*size(indices)[1], 3)
    for i=1:size(indices)[1]
        counter = 0
        for kk=-1:1, jj=-1:1, ii=-1:1
            all_indices[counter+(i-1)*27+1, :] = [indices[i, 1]+ii, 
                                                  indices[i, 2]+jj, 
                                                  indices[i, 3]+kk]
            counter+=1
        end
    end
    return all_indices
end
```




    process_indices (generic function with 1 method)



For an individual catalogue we would have a set of indices from the catalogue as


```julia
catalogue_indices = floor.(cat(dims=3, ones(num_catalogues, max_catalogue_length) / L0 * N0,
                                ones(num_catalogues, max_catalogue_length) / L1 * N1,
                                ones(num_catalogues, max_catalogue_length) / L2 * N2))
processed_catalogue_indices = process_indices(catalogue_indices[1, :, :])
processed_catalogue_indices = reshape(processed_catalogue_indices, (1, size(processed_catalogue_indices)...));
```


```julia
catalogue_indices = tf.Variable(
    processed_catalogue_indices,
    trainable=false,
    name="catalogue_indices")
```




    Variable{Int64}(<Tensor catalogue_indices:1 shape=(1, 81000, 3) dtype=Int64>, <Tensor catalogue_indices/Assign:1 shape=(1, 81000, 3) dtype=Int64>)



Now when we run the inference we need to collect the correct catalogue from the stored tensors. This can be done using a placeholder.


```julia
catalogue = tf.placeholder(
    Int64,
    shape=Int64[1],
    name="catalogue")
```




    <Tensor catalogue:1 shape=(1) dtype=Int64>



So now we can grab the density field at the indices in a particular halo catalogue and then reshape it into the 3x3x3 blocks for every halo all at the same time.


```julia
catalogue_δ = tf.reshape(
    tf.gather_nd(
        δ_LPT, 
        tf.gather_nd(catalogue_indices, catalogue)),
    Int64[max_catalogue_length, 3, 3, 3, 1],
    name="catalogue_density_LPT")
```




    <Tensor catalogue_density_LPT:1 shape=(3000, 3, 3, 3, 1) dtype=Float64>



Like with the LPT field, this is passed through the neural physical engine and then reshaped into a flat tensor with the length of the halo catalogue.


```julia
catalogue_δ_NPE = tf.reshape(
    convolution(catalogue_δ), 
    Int64[max_catalogue_length, 1],
    name="catalogue_density_NPE")
```




    <Tensor catalogue_density_NPE:1 shape=(3000, 1) dtype=Float64>



Now this can be masked correctly according to the selection effects mask corresponding to each halo. The correct mask is first applied to the halo masses.


```julia
this_mask = tf.identity(
    tf.gather_nd(
        mask,
        catalogue),
    name="this_mask")
masked_u = tf.boolean_mask(
    tf.gather_nd(
        catalogue_u,
        catalogue),
    this_mask,
    name="masked_u")
```




    <Tensor masked_u/Gather_2:1 shape=(?) dtype=Float64>



And we get the masked parameter values given the density environment using


```julia
catalogue_α, catalogue_μ, catalogue_σ = build_network(
        tf.boolean_mask(catalogue_δ_NPE, this_mask),
        layers, 
        distributions)
catalogue_α = tf.identity(catalogue_α, name="catalogue_density_amplitude")
catalogue_μ = tf.identity(catalogue_μ, name="catalogue_density_mean")
catalogue_σ = tf.identity(catalogue_σ, name="catalogue_density_covariance")
```




    <Tensor catalogue_density_covariance:1 shape=(?, 2) dtype=Float64>



We can evaluate the Gaussian at the masses in the halo catalogue with the parameters and the masked halo masses, and then summing over the Gaussians gives us the mixture density.


```julia
catalogue_g = tf.identity(
    gaussian(catalogue_α, 
             catalogue_μ, 
             catalogue_σ, 
             masked_u),
    name="catalogue_density_gaussian")
catalogue_n = tf.reduce_sum(catalogue_g, axis=2, name="catalogue_density_mixture")
```




    <Tensor catalogue_density_mixture:1 shape=(?) dtype=Float64>



The likelihood is given by the sum of the negative logarithm of mixture density plus the integral over the masses we calculated earlier (normalised by the grid size).


```julia
likelihood = tf.reduce_sum(
    subtract(
        tf.multiply(
            V,
            integral),
        tf.log(catalogue_n)),
    name="likelihood")
```




    <Tensor likelihood:1 shape=() dtype=Float64>



We need to include our prior which is a Gaussian of width `bias_width` over all of the parameters


```julia
prior = tf.multiply(
    divide(Float64(0.5),
           tf.square(bias_prior)),
    tf.reduce_sum(tf.square(p)),
    name="prior")
```




    <Tensor prior:1 shape=() dtype=Float64>



So the cost function to evaluate is


```julia
Λ = tf.add(
    likelihood,
    prior,
    name="loss")
```




    <Tensor loss:1 shape=() dtype=Float64>



Now for `BORG` we need the gradients with respect to the LPT field and the gradients with respect to all of the parameters of the neural bias model.


```julia
adgrad = tf.gradients(Λ, δ_LPT)
wgrad = tf.gradients(Λ, p)
```

We now have all the ingredients we need to run the HMC sampling of the neural bias model with `BORG`. For access to the mainframe of `BORG`, please contact the Aquila consortium
