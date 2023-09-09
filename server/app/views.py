from flask import Flask, request
import numpy as np

#######################################################################################
from app import app
from common.pereferences import DEBUG, PORT, HOST, THREADED

@app.route('/')
@app.route('/index')
def source():
    return "SERVER AHAAHAHAHAHAHAHHAHAHAAHHAHAHAHA"


def main():
    app.run(HOST, PORT, debug=DEBUG, threaded=THREADED)

if __name__ == '__main__':
    main()