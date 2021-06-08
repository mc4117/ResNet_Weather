The python files in this folder are used to train individual ResNets to produce a probabilistic weather forecast. This is done by transforming the data to categorical and then using a SoftMax layer. Below we give a brief description of each file and the section it corresponds to in the accompanying paper.

    orig_no_dr_z.py/orig_no_dr_t.py - training ResNet using only Z500 and T850 - Section 2.2
    continuous_72.py - training on continuous data (using only Z500 and T850) for comparison - Section 2.2
    orig_dr_z.py/orig_dr_t.py - training ResNet using only Z500 and T850 with dropout - Section 2.2.1
    level_analysis_t.py/level_analysis_z.py - determining optimum variables and levels - Sections 2.3.1
    indiv_member_train*.py - used to generate inputs for stacked neural network and for determining optimum number of residual blocks - Sections 3   
    stack_train* - used to train stacked neural networks and to make final accuracy estimates for our neural network approach - Section 3

The main results of are collated in results_summary.ipynb and `outputs_example_z.ipynb` and `outputs_example_t.ipynb` show examples of outputs produced by the neural network, in this case for Storm Ophelia in October 2017 (see Section 3 of accompanying paper).

