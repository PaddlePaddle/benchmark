import platform
import os
import subprocess

def get_incrementapi():
    '''
    this function will get the apis that difference between API_DEV.spec and API_PR.spec.
    '''
    def get_api_md5(path):
        api_md5 = {}
        with open(path) as f:
            for line in f.readlines():
                api = line.split(' ', 1)[0]
                md5 = line.split(' ', 1)[1]
                api_md5[api] = md5
        return api_md5

    dev_api = get_api_md5('scripts/API_DEV.spec')
    pr_api = get_api_md5('scripts/API_PR.spec')
    with open('dev_pr_diff_api.spec', 'w') as f:
        for key in pr_api:
            if key in dev_api:
                if dev_api[key] != pr_api[key]:
                    f.write(key)
                    f.write('\n')
            else:
                f.write(key)
                f.write('\n')

def run_diff_api(API_spec):
    results = []
    with open(API_spec) as f:
        for line in f.readlines():
            api_py = line.split('/')[2].strip()
            api = line.split('/')[2].replace('.py', '').strip()
            parameter_str = '--task accuracy --framework paddle --json_file examples/%s.json --config_id 0 --run_with_executor True --check_output False --profiler none --backward False --use_gpu True --repeat 1 --log_level 0' %api
            parameter = parameter_str.split(' ')
            if platform.python_version()[0] == "2":
                cmd = ["python", api_py] + parameter 
            elif platform.python_version()[0] == "3":
                cmd = ["python3", api_py] + parameter 
            else:
                print("Error: fail to parse python version!")
                result = False
                exit(1)
            subprc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd="api/tests", env={"CUDA_VISIBLE_DEVICES": "1"})
            output, error = subprc.communicate()
            msg = "".join(output.decode(encoding='utf-8'))
            err = "".join(error.decode(encoding='utf-8'))
            if subprc.returncode != 0:
                print("\nSample code error found in ", api, ":\n")
                print("subprocess return code: ", str(subprc.returncode))
                print("Error Raised from Sample Code ", api, " :\n")
                print(err)
                print(msg)
                result = False
            else:
                result = True
            results.append(result)
    return results

print("API check -- Example Code")
print("sample_test running under python", platform.python_version())
get_incrementapi()
API_spec = 'dev_pr_diff_api.spec'
if os.path.getsize(API_spec) != 0:
    results = run_diff_api(API_spec)
    if False in results:
        print("Mistakes found in api test.")
        print("Please check api.py.")
        exit(1)
    else:
        print("Api test check is successful")
else:
    print("-----API_PR.spec is the same as API_DEV.spec-----")
