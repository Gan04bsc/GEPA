def json_encoder(o):
    try:
        return {**o}
    except:
        return repr(o)
