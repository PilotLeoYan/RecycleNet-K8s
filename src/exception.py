import sys
import traceback


class RecycleNetException(Exception):
    def __init__(
        self,
        error_message: str
    ) -> None:
        _, _, exc_tb = sys.exc_info()

        if exc_tb is None:
            self.message = f"Error: {error_message}"
        else:
            tb_details = traceback.extract_tb(exc_tb)
            self.message = f"Error in '{tb_details[-1].filename}', line {tb_details[-1].lineno}: {error_message}"

        super().__init__(self.message)

    def __str__(
        self
    ) -> str:
        return self.message
