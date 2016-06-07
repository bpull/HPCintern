#!/usr/bin/env python
# python_example.py
# Author: Ben Goodrich
#
# This is a direct port to python of the shared library example from
# ALE provided in doc/examples/sharedLibraryInterfaceExample.cpp
import sys
import h5py
import numpy as np
from random import randrange
from ale_python_interface import ALEInterface

if len(sys.argv) < 2:
  print 'Usage:', sys.argv[0], 'rom_file'
  sys.exit()

ale = ALEInterface()
h5f = h5py.File('data.h5', 'w')
ram0 = np.ndarray(shape=(1,), dtype=np.uint8)
act  = np.ndarray(shape=(1,), dtype=np.uint8)
rew  = np.ndarray(shape=(1,), dtype=np.uint8)
ram1 = np.ndarray(shape=(1,), dtype=np.uint8)

# Get & Set the desired settings
ale.setInt('random_seed', int(sys.argv[2]))

# Set USE_SDL to true to display the screen. ALE must be compilied
# with SDL enabled for this to work. On OSX, pygame init is used to
# proxy-call SDL_main.
USE_SDL = False
if USE_SDL:
  if sys.platform == 'darwin':
    import pygame
    pygame.init()
    ale.setBool('sound', False) # Sound doesn't work on OSX
  elif sys.platform.startswith('linux'):
    ale.setBool('sound', True)
  ale.setBool('display_screen', True)

# Load the ROM file
ale.loadROM(sys.argv[1])

# Get the list of legal actions
legal_actions = ale.getLegalActionSet()

# Play 10 episodes
for episode in xrange(int(sys.argv[3])):

    fours = 0
    counter = 0
    cur_reward = 0
    total_reward = 0
    start = 1

    while not ale.game_over():
        if fours % 4 == 0:
            ram_size = ale.getRAMSize()
            ram = np.zeros((ram_size),dtype=np.uint8)
            ale.getRAM(ram)
            a = legal_actions[randrange(len(legal_actions))]

            if start:
                ram0 = ram
                act = a
            else:
                ram0 = np.vstack([ram0, ram])
                act = np.vstack([act, a])


        # Apply an action and get the resulting reward
        reward = ale.act(a);

        total_reward += reward
        cur_reward += reward

        #print ram in binary
        if fours % 4 == 3:
            ale.getRAM(ram)
            if start:
                ram1 = ram
                rew = cur_reward
                start = 0
            else:
                rew = np.vstack([rew, cur_reward])
                ram1 = np.vstack([ram1, ram])

            cur_reward = 0
            counter=counter+1

        fours=fours+1

    print 'Episode', episode, 'ended with score:', total_reward
    ale.reset_game()

dset0 = h5f.create_dataset("s", data=ram0)
dset1 = h5f.create_dataset("a", data=act)
dset2 = h5f.create_dataset("r", data=rew)
dset3 = h5f.create_dataset("sp", data=ram1)
