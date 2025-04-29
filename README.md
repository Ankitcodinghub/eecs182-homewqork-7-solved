# eecs182-homewqork-7-solved
**TO GET THIS SOLUTION VISIT:** [EECS182 Homewqork 7 Solved](https://www.ankitcodinghub.com/product/eecs182-solved-4/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;116349&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;2&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (2 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;EECS182 Homewqork 7  Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (2 votes)    </div>
    </div>
1. Auto-encoder : Learning without Labels

So far, with supervised learning algorithms we have used labelled datasets D = {X,y} to learn a mapping fθ from input x to labels y. In this problem, we now consider algorithms for unsupervised learning, where we are given only inputs x, but no labels y. At a high-level, unsupervised learning algorithms leverage some structure in the dataset to define proxy tasks, and learn representations that are useful for solving downstream tasks.

Autoencoders present a family of such algorithms, where we consider the problem of learning a function fθ : Rm → Rk from input x to a intermediate representation z of x (usually k ≪ m). For autoencoders, we use reconstructing the original signal as a proxy task, i.e. representations are more likely to be useful for downstream tasks if they are low-dimensional but encode sufficient information to approximately reconstruct the input. Broadly, autoencoder architectures have two modules:

• An encoder fθ : Rm → Rk that maps input x to a representation z.

• A decoder gϕ : Rk → Rm that maps representation z to a reconstruction xˆ of x.

In such architectures, the parameters (θ,ϕ) are learnable, and trained with the learning objective of minimizing the reconstruction error using gradient descent.

N

,

i=1

xˆ = gϕ(fθ(x))

Note that above optimization problem does not require labels y. In practice however, we would want to use the pretrained models to solve the downstream task at hand, e.g. classifying MNIST digits. Linear Probing is one such approach, where we fix the encoder weights, and learn a simple linear classifier over the features z = encoder(x).

(a) Designing AutoEncoders

Please follow the instructions in this notebook. You will train autoencoders, denoising autoencoders, and masked autoencoders on a synthetic dataset and the MNIST dataset. Once you finished with the notebook,

• Download submission_log.json and submit it to “Homework 7 (Code)” in Gradescope.

• Answer the following questions in your submission of the written assignment:

(i) Show your visualization of the vanilla autoencoder with different latent representation sizes.

(ii) Based on your previous visualizations, answer this question: How does changing the latent representation size of the autoencoder affect the model’s performance in terms of reconstruction accuracy and linear probe accuracy? Why?

(b) PCA &amp; AutoEncoders

In the case where the encoder fθ,gϕ are linear functions, the model is termed as a linear autoencoder. In particular, assume that we have data xi ∈ Rm and consider two weight matrices: an encoder W1 ∈ Rk×m and decoder W2 ∈ Rm×k (with k &lt; m). Then, a linear autoencoder learns a lowdimensional embedding of the data X ∈ Rm×n (which we assume is zero-centered without loss of generality) by minimizing the objective,

(1)

We will assume are the k largest eigenvalues of XX⊤. The assumption that the σ1,…,σk are positive and distinct ensures identifiability of the principal components, and is common in this setting. Therefore, the top-k eigenvalues of X are S = diag(σ1,…,σk), with corresponding eigenvectors are the columns of Uk ∈ Rm×k. A well-established result from (Baldi &amp; Hornik, 1989) shows that principal components are the unique optimal solution to linear autoencoders (up to sign changes to the projection directions). In this subpart, we take some steps towards proving this result.

(i) Write out the first order optimality conditions that the minima of Eq. 1 would satisfy.

(ii) Show that the principal components Uk satisfy the optimality conditions outlined in (i).

2. Self-supervised Linear Autoencoders

We consider linear models consisting of two weight matrices: an encoder W1 ∈ Rk×m and decoder W2 ∈ Rm×k (assume 1 &lt; k &lt; m). The traditional autoencoder model learns a low-dimensional embedding of the n points of training data X ∈ Rm×n by minimizing the objective,

(2)

We will assume are the k + 1 largest eigenvalues of . The assumption that the σ1,…,σk are positive and distinct ensures identifiability of the principal components.

Consider an ℓ2-regularized linear autoencoder where the objective is:

. (3)

where ∥ · ∥2F represents the Frobenius norm squared of the matrix (i.e. sum of squares of the entries).

(a) You want to use SGD-style training in PyTorch (involving the training points one at a time) and selfsupervision to find W1 and W2 which optimize (3) by treating the problem as a neural net being trained in a supervised fashion. Answer the following questions and briefly explain your choice:

(i) How many linear layers do you need?

□ 0

□ 1

□ 2

□ 3

(ii) What is the loss function that you will be using?

□ nn.L1Loss

□ nn.MSELoss

□ nn.CrossEntropyLoss

(iii) Which of the following would you need to optimize (3) exactly as it is written? (Select all that are needed)

□ Weight Decay

□ Dropout

□ Layer Norm □ Batch Norm

□ SGD optimizer

(b) Do you think that the solution to (3) when we use a small nonzero λ has an inductive bias towards finding a W2 matrix with approximately orthonormal columns? Argue why or why not?

(Hint: Think about the SVDs of . You can assume that if a k × m or m × k matrix has all k of its nonzero singular values being 1, then it must have orthonormal rows or columns. Remember that the Frobenius norm squared of a matrix is just the sum of the squares of its singular values. Further think about the minimizer of . Is it unique?)

3. Justifying Scaled-Dot Product Attention

Suppose q,k ∈ Rd are two random vectors with q,k N(µ,σ2I), where µ ∈ Rd and σ ∈ R+. In other words, each component qi of q is drawn from a normal distribution with mean µ and stand deviation σ, and the same if true for k.

(a) Define E[qT k] in terms of µ,σ and d.

(b) Considering a practical case where µ = 0 and σ = 1, define Var(qT k) in terms of d.

(c) Continue to use the setting in part (b), where µ = 0 and σ = 1. Let s be the scaling factor on the dot

T product. Suppose we wantto be 0, and Varto be σ = 1. What should s be in terms of d? 4. Argmax Attention

(a) Perform argmax attention with the following keys and values:

Keys:

Corresponding Values: using the following query:  

1

q = 1

 

2

What would be the output of the attention layer for this query?

Hint: For example, argmax([1,3,2]) = [0,1,0]

(b) Note that instead of using softmax we used argmax to generate outputs from the attention layer. How does this design choice affect our ability to usefully train models involving attention?

(Hint: think about how the gradients flow through the network in the backward pass. Can we learn to improve our queries or keys during the training process?)

5. Kernelized Linear Attention

Tl(x) = fl(Al(x) + x). (4)

The function fl(·) transforms each feature independently of the others and is usually implemented with a small two-layer feedforward network. Al(·) is the self attention function and is the only part of the transformer that acts across sequences.

We now focus on the the self attention module which involves softmax. The self attention function Al(·) computes, for every position, a weighted average of the feature representations of all other positions with a weight proportional to a similarity score between the representations. Formally, the input sequence x is projected by three matrices WQ ∈ RF×D, WK ∈ RF×D and WV ∈ RF×M to corresponding representations Q, K and V . The output for all positions, Al(x) = V ′, is computed as follows,

Q = xWQ,K = xWK,V = xWV ,

(5)

′

Al(x) = V = softmax

Note that in the previous equation, the softmax function is applied rowwise to QKT . Following common terminology, the Q, K and V are referred to as the “queries”, “keys” and “values” respectively.

Equation 5 implements a specific form of self-attention called softmax attention where the similarity score is the exponential of the dot product between a query and a key. Given that subscripting a matrix with i returns the i-th row as a vector, we can write a generalized attention equation for any similarity function as follows,

. (6)

sim

Equation 6 is equivalent to equation 5 if we substitute the similarity function with simsoftmax(q,k) = . This can lead to

. (7)

For computing the resulting self-attended feature Al(x) = V ′, we need to compute all in equation 7.

(a) Identify the conditions that needs to be met by the sim function to ensure that Vi in Equation 6 remains finite (the denominator never reaches zero).

(b) The definition of attention in equation 6 is generic and can be used to define several other attention implementations.

(i) One potential attention variant is the “polynomial kernel attention”, where the similarity function as sim(q,k) is measured by polynomial kernel K . Considering a special case for a “quadratic kernel attention” that the degree of “polynomial kernel attention” is set to be 2, derive the sim(q,k) for “quadratic kernel attention”. (NOTE: any constant factor is set to be 1.) .

(ii) One benefit of using kernelized attention is that we can represent a kernel using a feature map ϕ(·) . Derive the corresponding feature map ϕ(·) for the quadratic kernel.

(iii) Considering a general kernel attention, where the kernel can be represented using feature map that K(q,k) = (ϕ(q)T ϕ(k)), rewrite kernel attention of equation 6 with feature map ϕ(·).

(c) We can rewrite the softmax attention in terms of equation 6 as equation 7. For all the Vi′ (i ∈ {1,…,N}), derive the time complexity (asymptotic computational cost) and space complexity (asymptotic memory requirement) of the above softmax attention in terms of sequence length N,

D and M.

NOTE: for memory requirement, we need to store any intermediate results for backpropagation, including all Q,K,V

(d) Assume we have a kernel K as the similarity function and the kernel can be represented with a feature map ϕ(·), we can rewrite equation 6 with sim in part (b). We can then further simplify it by making use of the associative property of matrix multiplication to

. (8)

Note that the feature map ϕ(·) is applied row-wise to the matrices Q and K.

Considering using a linearized polynomial kernel ϕ(x) of degree 2, and assume M ≈ D, derive the computational cost and memory requirement of this kernel attention as in (8).

6. Debugging DNNs (Optional)

(a) Your friends want to train a classifier for a new app they’re designing. They implement a deep convolutional network and train models with two configurations: a 20 layer model and a 56 layer model. However, they observe the following training curves and are surprised that the 20-layer network has better training as well as test error.

Figure 1: Training deep networks on CIFAR10

What are the potential reasons for this observation? Are there changes to the architecture design that could help mitigate the problem?

(b) You and your teammate want to compare batch normalization and layer normalization for the ImageNet classification problem. You use ResNet-152 as a neural network architecture. The images have input dimension 3 × 224 × 224 (channels, height, width). You want to use a batch size of 1024; however, the GPU memory is so small that you cannot load the model and all 1024 samples at once — you can only fit 32. Your teammate proposes using a gradient-accumulation algorithm:

Gradient accumulation refers to running the forward and backward pass of a model a fixed number of steps (accumulation_steps) without updating the model parameters, while aggregating the gradients. Instead, the model parameters are updated every (accumulation_steps). This allows us to increase the effective batch size by a factor of accumulation_steps.

You implement the algorithm in PyTorch as:

model.train() optimizer.zero_grad() for i, (inputs, labels) in enumerate(training_set): predictions = model(inputs) loss = loss_function(predictions, labels) loss = loss / accumulation_steps loss.backward() if (i+1) % accumulation_steps == 0:

optimizer.step() optimizer.zero_grad() Note that the .backward() operator in PyTorch implicitly keeps accumulating the gradient for all the parameters, unless zero’d out with an optimizer.zero_grad() call.

Before running actual experiments, your friend suggests that you should test whether the gradient accumulation algorithm is implemented correctly. To do so, you collect the output logits (i.e. the outputs of the last layer) from two models — ResNet-152, one with batchnorm and the other with layernorm — using different combinations of batch sizes and the number of accumulation steps that keep the effective batch size to 32. h

Note that the effective batch size is product of the batch size and accumulation_steps. In other words, the possible combinations for effective batch-size 32 are:

i (batch_size, accumulation_steps) = (1, 32), (2, 16), (4, 8), (8, 4), (16, 2), (32, 1).

Here, the (32,1) combination is the approach without the “gradient accumulation” trick, and we want to see whether the others agree with this.

On running these tests, you observe that one of models: either with batchnorm or with layernorm, doesn’t pass the test. Which one do you expect to not pass the test and why?

(c) You are training a CIFAR model and observe that the model is diverging (instead of the training loss decreasing over iterations). Debug the pseudocode and give a correction that you believe would actually result in reasonable convergence during training.

(Note: You can assume that the datasets are loaded correctly, model is trained with SGD optimizer with learning rate= 0.001, batchsize= 100)

(HINT: Ideas from the previous part of this question might be relevant.)

model . t r a i n ( ) optimizer . zero_grad ( ) for ( inputs , l a b e l s ) in t r a i n i n g _ s e t : p r e d i c t i o n s = model ( inputs ) loss = loss_fn ( predictions , l a b e l s ) loss . backward ( ) optimizer . step ( )

Figure 2: Training loss for CIFAR10

7. Homework Process and Study Group

We also want to understand what resources you find helpful and how much time homework is taking, so we can change things in the future if possible.

(a) What sources (if any) did you use as you worked through the homework?

(b) If you worked with someone on this homework, who did you work with?

List names and student ID’s. (In case of homework party, you can also just describe the group.)

(c) Roughly how many total hours did you work on this homework? Write it down here where you’ll need to remember it for the self-grade form.

References

Baldi, P. and Hornik, K. Neural networks and principal component analysis: Learning from examples without local minima. Neural networks, 2(1):53–58, 1989.

Luong, M.-T., Pham, H., and Manning, C. D. Effective approaches to attention-based neural machine translation. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pp. 1412–1421, 2015.
