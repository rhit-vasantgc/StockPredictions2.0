#subprocess is op
#this allows you to call other programs in this program
#will have to utilize the subprocess module later most likely
#need the full path if the file isn't in the same directory
#source link: 
#https://www.w3docs.com/snippets/python/how-can-i-make-one-python-file-run-another.html#:~:text=To%20run%20one%20Python%20file,function%20or%20the%20subprocess%20module.&text=This%20will%20execute%20the%20code,in%20the%20main.py%20file.&text=This%20will%20run%20the%20other.py%20script%20as%20a%20separate%20process.
import subprocess
subprocess.run(["python","scrapper.py"])
subprocess.run(["python","normalization.py"])
subprocess.run(["python","norm_diff.py"])