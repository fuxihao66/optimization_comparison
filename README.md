## THIS DOCUMENT IS ABOUT THE STRUCTURE OF CODES
### In cnn
1. utils
    * cnn_base:  the class of cnn model
    * plot_test: plot graph
2. experiments
    * cnn_first: the first experiment in the paper
    * cnn_lr_*:  the learning rate experiments
    * cnn_batch_all: run the batch size experiments
    * cnn_final_*: the final experiment in the paper(every method with 2 best configurations) 
### In softmax
1. utils
    * softmax_zeros:  softmax models with different initials
    * plot_test: plot graph
2. experiments
    * softmax_first: the first experiment in the paper
    * softmax_lr_all: run the lr experiments
    * softmax_batch_all: run the batch size experiments
    * softmax_init_all: run the initialization experiments
    * softmax_final_*: run the final experiments in the paper
