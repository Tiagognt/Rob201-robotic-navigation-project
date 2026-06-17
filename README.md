# Robotic navigation project conducted during the ROB201 class at ENSTA Paris

This git repository contain the source code of the implementation of SLAM algorithm in the place-bot simulator.

To start exloring the ROB201 project, you will have to install this repository (which will install required dependencies).

# Setup

Start by cloning this repository in the desired location

~~~
  cd path/to/your/location # replace the "/" by "\" if you are on windows
  git clone https://github.com/Tiagognt/Rob201-robotic-navigation-project.git
~~~

get into the `Rob201-robotic-navigation-project` folder and run the folowing comands

~~~
  python3 -m venv .venv &&
  source .venv/bin/activate
  pip install -r requirements.txt
~~~

Once this installation is complete you are ready to run the project

# Launching

To launch the main simulation go to `/tp_rob201` and run 
~~~
  python3 main.py
~~~
And press `q` to quit.

# About Place-Bot

This project use the **Place-Bot** simulator: [**Place-Bot** GitHub repository](https://github.com/emmanuel-battesti/place-bot) wich is automaticly installed during the installation process. To undersand more deeply the architecture of this project it is strongly recommended to read the [*Place-Bot* documentation](https://github.com/emmanuel-battesti/place-bot#readme).





