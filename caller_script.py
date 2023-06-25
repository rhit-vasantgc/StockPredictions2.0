#subprocess is op
#this allows you to call other programs in this program
#will have to utilize the subprocess module later most likely
#need the full path if the file isn't in the same directory
#source link: 
#https://www.w3docs.com/snippets/python/how-can-i-make-one-python-file-run-another.html#:~:text=To%20run%20one%20Python%20file,function%20or%20the%20subprocess%20module.&text=This%20will%20execute%20the%20code,in%20the%20main.py%20file.&text=This%20will%20run%20the%20other.py%20script%20as%20a%20separate%20process.
import subprocess
# with open("scrapper.py") as scrap:
#     exec(scrap.read())
# with open("normalization.py") as norm:
#     exec(norm.read())
# with open("norm_diff.py") as ndiff:
#     exec(ndiff.read())
# with open("graphing.py") as graphing:
#     exec(graphing.read())
# with open("analysism2.py") as analysism2:
#     exec(analysism2.read())
# with open("model.py") as model:
#     exec(model.read())

subprocess.run(["python","scrapper.py"],shell=True)
subprocess.run(["python","normalization.py"],shell=True)
subprocess.run(["python","norm_diff.py"],shell=True)
subprocess.run(["python","graphing.py"],shell=True)
subprocess.run(["python","analysism2.py"],shell=True)
subprocess.run(["python","model.py"],shell=True)
#denormalizer.py should be run MANUALLY after all of this is done (manually because of concurrent mods)