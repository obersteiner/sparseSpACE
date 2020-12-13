# make the hook runnable
chmod +x git_hooks/pre-commit

# configure git to use the new hook - it will stay with the repository
git config core.hooksPath "./git_hooks"

# install sparseSpACE
pip3 install -e .
