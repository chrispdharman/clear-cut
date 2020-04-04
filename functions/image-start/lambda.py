print('Started processing image ...')
def handler(event, context):
    print('[Handling event]')
    print('... image processing complete!')
    return {'description': 'Hello!'}
