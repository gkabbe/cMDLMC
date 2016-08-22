from mdkmc import version_hash

def print_version():
    print(version_hash.commit_hash)
    print(version_hash.commit_date)
    print(version_hash.commit_message)