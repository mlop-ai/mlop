import json
import logging


def make_compat_status_v1(data, settings):
    return json.dumps(
        {
            "runId": settings._op_id,
            "runName": settings._op_name,
            "projectName": settings.project,
            "metadata": json.dumps(data),
        }
    ).encode()


def make_compat_meta_v1(meta, settings):
    return json.dumps(
        {
            "runId": settings._op_id,
            # "runName": settings._op_name,
            # "projectName": settings.project,
            "logName": meta,  # TODO: better aggregate
        }
    ).encode()


def make_compat_data_v1(data, timestamp, step):
    line = [
        json.dumps(
            {
                "time": int(timestamp * 1000),  # convert to ms
                "step": int(step),
                "data": data,
            }
        )
    ]
    return ("\n".join(line) + "\n").encode("utf-8")


def make_compat_file_v1(file, timestamp, step):
    batch = []
    for k, fl in file.items():
        for f in fl:
            i = {
                "fileName": f"{f._name}{f._ext}",
                "size": f._size,
                "fileType": f._ext[1:],
                "logName": k,
                "step": step,
            }
            batch.append(i)
    return json.dumps({"files": batch}).encode()


def make_compat_storage_v1(f, fl):
    # workaround for lack of file ident on server side
    for i in fl:
        if next(iter(i.keys())) == f"{f._name}{f._ext}":
            return next(iter(i.values()))
    return None


def make_compat_message_v1(level, message, timestamp, step):
    line = [
        json.dumps(
            {
                "time": int(timestamp * 1000),  # convert to ms
                "message": message,
                "lineNumber": step,
                "logType": "INFO" if level == logging.INFO else "ERROR",
            }
        )
    ]
    return ("\n".join(line) + "\n").encode("utf-8")
