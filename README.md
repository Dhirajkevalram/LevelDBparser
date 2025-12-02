

A powerful GUI-based tool for parsing and analyzing LevelDB databases. 

## Before You Begin – Install WSL on Windows
To run this tool on Windows with full GUI support, you must first install WSL (Windows Subsystem for Linux).

STEP 1) Open **PowerShell** as Administrator  

Press Start → type *PowerShell* → Right-click → **Run as Administrator**

STEP 2)  Install WSL + Ubuntu  

        ```powershell
        wsl --install


STEP 3) Restart your PC

STEP 4) Complete Ubuntu setup
       
        Create: Linux Username and  Linux Password

STEP 5) Make sure WSL2 is active:

        wsl --set-default-version 2
  
STEP 6) Navigate to your folder: 

        cd ~/path/to/your/project 

STEP 7) Activate virtual environment

        source venv/bin/activate

STEP 8) Install all dependencies using 
         
          python3 -m pip install --upgrade pip
         pip install plyvel python-snappy pandas PySide6.
              
 STEP 9) Run the Python Script  using
                        
        python3 LevelDBparser.py

You are now ready to run Linux GUI forensic tools inside Windows.
