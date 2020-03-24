import logging

def get_log(name,
            filename='log.log',
            level=logging.INFO,
            # log_format='%(asctime)s - %(levelname)s - %(message)s',
            log_format="%(asctime)s - %(name)s - %(levelname)s: %(message)s",
            log_datefmt='"%Y-%m-%d %H:%M:%S"',
            ):
    '''A simple log function.

    Args:
        filename: string
            the name of `.log` file. `default:log.log`
        level:
            such as `logging.INFO`
        name:
            for `getLogger(name)`

    Return:
        return the Logger.
    '''
    logging.basicConfig(
            level=level,
            format=log_format,
            datefmt=log_datefmt,
            filename=filename,
    )
    logger = logging.getLogger(name)
    return logger