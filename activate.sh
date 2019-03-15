#! /bin/bash

ENV="OpenAi";


ENVS=$(conda env list | awk '{print $ENV}' )
if [[ $ENVS = *"$ENV"* ]]; then
   source activate $ENV
else 
   echo "INSTALLING"
   conda env create -f environment.yml
   source activate $ENV
   pip install -r requirements.txt
fi;

echo Ezz;
