# %%

from viconnexusapi import ViconNexus

vicon = ViconNexus.ViconNexus()

# %%

vicon.IsConnected()
# %%

vicon.GetTrialName()

# %%
vicon.GetSubjectNames()
# %%

vicon.OpenTrial("D:/HPL/t20/NUSHUcalib_01", 20)

# %%
