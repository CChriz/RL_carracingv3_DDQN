--------------------
Installation & Setup
--------------------

we took the following steps in setting up the environment and producing our results:

Install anaconda: https://www.anaconda.com/download

Create anaconda environment
- navigate to Anaconda Prompt (NOT command prompt) and execute the following scripts:
	conda create -n gymenv
	conda activate gymenv
	conda install python=3.11
	conda install swig
	pip install gymnasium[box2d]
(you may need to install C++ build tools from the link in the error message if not installed before. Ensure you select Desktop Developer with C++ then try again):   
	pip install gymnasium[box2d]


we used Visual Studio Code for developing our code, to run the code:
1. Open our code folder in Visual Studio Code
2. go to [Select Interpreter] and choose "gymenv" - the one created with conda (Ctrl+Shift+P)
3. go to [Terminal Select Default Profile] and choose "Command Prompt" instead of "PowerShell".
4. go to [Files] and open Terminal (command prompt) within VS Code
5. Execute:
	conda activate gymenv


You should now be able to execute validation/training/evaluation scripts:

- to validate the environment can be created and rendered successfully, run the "validation.py" file

- to play the rendered environment yourself (human control), run the "youplay.py" file	


we recommend following the following YouTube video for installation and setup of the Box2D CarRacing-v3 environment:
https://www.youtube.com/watch?time_continue=429&v=gMgj4pSHLww&embeds_referring_euri=https%3A%2F%2Fdiscord.com%2F&source_ve_path=MjM4NTE


---------------
Files
---------------

Single Frame DDQN agent:
- DDQN_train.py : script to train a DDQN agent using single frames

- DDQN_load.py : script to load and evaluate a pre-trained DDQN agent

Saved DDQN (single frame) models:
- naming convention: "ddqn_model<episode>.h5" : pre-trained model weights at <episode>


Stacked Frame DDQN agent:
- DDQN_stacking_train.py : script to train a DDQN agent using frame stacking

- DDQN_stacking_load.py : script to load and evaluate a pre-trained DDQN agent with frame stacking

Saved DDQN (stacked frames) models:
- naming convention: "ddqn_checkpoint_<episode>.h5" : pre-trained model weights at <episode>


---------------
Usage
---------------

To train the single frame DDQN agent, run the DDQN_train.py file
- by default this trains the agent with the following attributes:
        env_name='CarRacing-v3',
        continuous=True, # set to continuous but mapped to discrete with DISCRETE_ACTIONS
        n_episodes=500, # total training episodes
        max_steps=1000, # max steps per episode
        batch_size=64, # replay batch size
        gamma=0.99, 
        epsilon_start=1.0, # initial epsilon value (for epsilon-greedy action choice)
        epsilon_min=0.05, # min epsilon value 
        epsilon_decay=0.995, # epsilon decay rate
        update_target_freq=1000, # rate of TARGET net update
        render=False, # environment render (false by default for training)
        save_freq=10, # save model every 10 episodes
        save_path='ddqn_model.h5' # file name to save model
upon training completion, a graph of episode vs total reward is plotted.

To train the stacked frames DDQN agent, run the DDQN_stacking_load.py file
- by default this trains the agent with the following attributes:
        env_name='CarRacing-v3',
        continuous=True, # continuous space but actions mapped to discrete in DISCRETE_ACTIONS
        stack_size=4, # number of stacked consecutive frames (used 4)
        frame_skip=4, # number of frames skipped per decision (used 4, same as stacked consecutive frames)
        n_episodes=2000, # number of episodes to train agent for
        max_steps=250, # max step per episode - only 250 since each decision repeats action for 4 frames (~1000 frames = max environment step)
        batch_size=64, # batch size for replay buffer
        gamma=0.99,
        epsilon_start=1.0, # initial epsilon value (for epsilon-greedy action choice)
        epsilon_min=0.05, # min epsilon value 
        epsilon_decay=0.997, # epsilon decay rate
        update_target_freq=500, # TARGET network updates per _ steps (default: 500)
        render=False, # if render environment (false by default for training)
        save_freq=20, # save model checkpoints every 50 episodes
        checkpoint_prefix='ddqn_checkpoint' # model save path prefix - suffix: episode, e.g. _420
upon training completion, a graph of episode vs total reward is plotted.

To load and evaluate a pre-trained single frame DDQN agent, run the DDQN_load.py file
- by default this loads the _ model and renders its performance for 10 episodes. Actions chosen at each step and rewards achieved are printed.

To load and evaluate a pre-trained stacked frames DDQN agent, run the DDQN_stacking_load.py file
- by default this loads the _ model and renders its performance for 10 episodes. Actions chosen at each step and rewards achieved are printed.


explanations & instructions for Modifications 
(i.e. Training Hyperparameters: max episodes, max steps per episode, epsilon decay, etc.)
(i.e. Evaluation: model to load, number of episodes, etc.)
all indicated in detail in comments and docstrings in their respective code files.

Thank you for reading :)
