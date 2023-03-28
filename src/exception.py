import sys
from src.logger import logging

def error_details(error, sys_error:sys):

    _,_,exc_tb = sys_error.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = f"Error occured in python. File name: {file_name}, line number: {exc_tb.tb_lineno}, error message: {str(error)}"
    return error_message

class CustomException(Exception):

    def __init__(self, error_message, sys_error:sys):
        super().__init__(error_message)
        self.error_message = error_details(error_message,sys_error=sys_error)

    def __str__(self):
        return self.error_message
    

# if __name__=="__main__":

#     try:
#         a=1/0
#     except Exception as e:
#         logging.info("Divide by Zero")
#         raise CustomException(e,sys)