.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

CLIENT = {{ cookiecutter.client.lower().replace(' ', '_')}}
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT = {{ cookiecutter.project.lower().replace(' ', '_')}}
EXPERIMENT = {{ cookiecutter.experiment.lower().replace(' ', '_')}}
BUCKET = dif-$(CLIENT)-$(PROJECT)-bucket


#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
pip-all: test_environment
	$ python3 -m pip install -U pip setuptools wheel
	$ python3 -m pip install -r requirements.txt

## Make Dataset
data: requirements
	$ python3 src/data/make_dataset.py 
## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Upload Data to S3
move_data_to_s3:
	aws s3 sync $$HOME/$(CLIENT)-$(PROJECT)/$(EXPERIMENT)/data/raw s3://$(BUCKET)/$(EXPERIMENT)/raw-data/ 

## Download Data from S3
move_data_from_s3:
	aws s3 sync s3://$(BUCKET)/$(EXPERIMENT)/raw-data/ $$HOME/$(CLIENT)-$(PROJECT)/$(EXPERIMENT)/data/raw

## connecting mlflow to local sqlite database and s3 for artiffacts
mlflow_server:
	mlflow server -h 0.0.0.0:5000 \
	--backend-store-uri sqlite:///mydb.sqlite \
	--default-artifact-root s3://$(BUCKET)/mlflow-bucket	
	
## init DVC in S3
init_dvc_in_s3:
	dvc init --force
	dvc add data/ 
	git add .
	# git commit -m "Add raw data"
	dvc remote add -d storage s3://$(BUCKET)/dvcstore --force
	dvc push
	git push
	
## Set up python interpreter environment
environment:
	@echo ">>> Creating environment"
	python3 -m venv ../
	@echo ">>> activate: $ source ../bin/activate"

## Test python environment is setup correctly
test_environment:
	$ python3 test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
