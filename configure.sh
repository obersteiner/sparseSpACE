# make the hook runnable
chmod +x git_hooks/pre-commit

# configure git to use the new hook - it will stay with the repository
git config core.hooksPath "./git_hooks"

# install dependencies
pip3 install -r requirements.txt

# install newest chaospy version from github
./install_chaospy.sh
