import sys
from src.logger import logging

def error_message_detail(error, error_detail):
    exc_type, exc_value, exc_tb = error_detail
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_no = exc_tb.tb_lineno
    else:
        file_name = "Unavailable"
        line_no = "Unavailable"

    error_message = "Error occurred in python script [{0}] at line number [{1}] error message [{2}]".format(
        file_name, line_no, str(error))

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)
    
    def __str__(self):
        return self.error_message

    
#if __name__=="__main__":

#    try:
#        a=1/0
#    except Exception as e:
#        logging.info("Exception handling has started")
#        raise CustomException(e, sys)
