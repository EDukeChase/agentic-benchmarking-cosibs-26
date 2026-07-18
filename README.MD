To be able to use this repo for code, you need to have docker installed and the docker app running on you computer.
For docker installation, here is the link to download docker desktop: https://docs.docker.com/get-started/introduction/get-docker-desktop/

Now that you have docker installed, when you open up your IDE, you can have the option to open this project as a dev container. First open up the docker desktop app, then do that.

Alternative: Open the terminal. Make sure that you are in the project directory. If needed, open the project directory using "cd {PathToProjectDIrectory}". (do not paste the quotation marks into the terminal). Then, build the docker image using "docker build -t benchmark-image ." Finally, run the docker image using "docker run --rm -it -v "{PathToProjectDIrectory}:/app" benchmark-image bash (use the quotations this time)

Now you are in a fully virtual environment with all the dependencies you need for development, enjoy!

For actually working with the agents, the first thing you have to do is connect to the VPN (specifically the amc-vpn-ucdenver one)
Next, you have to type "az login" in the command line, and it will log you in automatically.
You can test if all of this works through the test.py file. Use the command "python test.py" (no quotations) to run the file. Do not use the run code button in VSCode.

For coding in this environment, PLEASE ALWAYS RUN "git pull origin main" BEFORE DOING ANY CODING. IF YOU FORGET, YOU WILL HAVE TO MOVE YOUR CODE THAT YOU JUST MADE OUT OF THE REPOSITORY TO SOMEWHERE SAFE AND RUN "git fetch origin" then "git reset --hard origin/main" then "git clean -fd."

Before exiting, make sure to commit and push any changes first, as exiting the docker container closes your development environment.
You can exit by typing "exit" in the command line, and by clicking in the bottom left of your vscode and clicking the "Close Remote Connection" option.