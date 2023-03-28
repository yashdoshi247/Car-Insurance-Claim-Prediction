import subprocess

#Calling training_pipeline
subprocess.check_call(['python','src/pipeline/training_pipeline.py'])

#Calling flask app
subprocess.check_call(['python','application.py'])