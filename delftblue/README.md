The only commands I recommend to run on the login node are `unzip/tar`, `cp/mv/mkdir` and other basic linux commands which need few resources. 
The rest (`python/conda/make/executing SharpVelvet/model counters`) should done via an actual DelftBlue job.

You will have to copy the `.json` files from the `example-configs` folder and update the path to match your setup. 
The `bash` scrips have the path setup as `/home/$USER`, which works out-of-the-box if you transfer the files to your DelftBlue home folder. 

I recommend setting up an `SSH key-pair` and saving the `<user>@<delftblue>` as an alias in `.ssh/config`. The instructions are present in the DelftBlue manual section linked in step 1. Ask me if you need help with that.  

1. [Login](https://doc.dhpc.tudelft.nl/delftblue/Remote-access-to-DelftBlue/) to DelftBlue.
2. [Transfer](https://doc.dhpc.tudelft.nl/delftblue/Data-transfer-to-DelftBlue/) the needed files to DelftBlue. For starting point, you can use the `ganak-v2.1-linux64` binary from [this link](https://msoos.org/private/ganak-v2.1-linux64), a copy of SharpVelvet, the `.json` (adapted) configs, and the 3 `.sh` scripts. To follow these instructions easily, transfer the aforementioned files to your home folder: `scp [-J <bastion>] -p <local_file> <delftblue>:~/`. The linked manual section describes this in more details, if needed.
3. Unzip archives if any are present: for `.zip` files use `unzip <archive.zip>` and for `.tar.xz` files : `tar -xf <archive.tar.xz>`.
4. To compile the generators and set up SharpVelvet deps using Conda, and then schedule the compilation job: `sbatch create_conda_env.sh` and then 'sbatch compile_generators.sh'.
5. Wait till the job finishes. This will take a while due to `conda` being slow, but it is needed only once. After submitting a job using `sbatch`, you will get a jobID. To check status of a job: `seff <jobid>`. More info, if needed, can be found in the [manual](https://doc.dhpc.tudelft.nl/delftblue/Slurm-trouble-shooting/) section.
6. In the same folder you ran `sbatch`, there will be a `slurm-<jobid>.out` file generated after the execution starts. You can open this file using your preferred editor (i.e. vim, nano, etc.) directly on DelftBlue. You can also transfer it to your local computer for further analysis if needed. Important: This file can be opened and read while the the job is still running, so you can check it's progress live.
7. Change the path in the `.json` configs (`SharpVelvet/counter_config_mc.json` and `SharpVelvet/generators/generator_config_mc.json`) to match your username, since Python does not allow the `$USER` shell variable. Now, you can generate instances using the `generate_velvet.sh` script: `sbatch generate_velvet.sh`. 
8. Wait till the job finishes.
9. Now you can finally run SharpVelvet on `ganak2.1` :). Just as in previous points run using `sbatch run_velvet.sh`.
10. After the SharpVelvet run has finished, you can see SharpVelvet's output in the corresponding `slurm-<jobid>.out` file, and the report should be generated in the usual folder in the `SharpVelvet` directory.

The relevant default parameters I used in the `generate` and `run` script are 12 CPU cores and 15 minutes maximum runtime. This enables instant scheduling with no wait times. When doing actual fuzzing, adjust as needed. However keep in mind that waiting time increases exponentially if requesting many (64/128/etc.) cores for longer periods (1/5/10hours) during periods of peak DelftBlue usage by fellow researchers.
For the compile script, since conda is slow, I used a default of `30` minutes. 
