import logging

from clear_cutter import ClearCut

logger = logging.getLogger()
logger.setLevel(logging.INFO)

logger.info('Started processing image ...')
def handler(event, context):
    logger.info('[Handling event]')
    clear_cut = ClearCut()

    #clear_cut.image_filename = 'Bob.jpeg'
    clear_cut.image_filename = event['filename']
    result = clear_cut.run()

    logger.info('... image processing complete!')
    return {'description': result}
