#!/bin/env python3

import os
import argparse
import shutil
import signal
import re
import pprint

ROOT_DIR = "."
TRASH_DIR = "backup/trash"
ARCHIVE_DIR = "backup/archive"
PART_ARCHIVE_DIR = "backup/parchive"

def ask(question, posstr="y", negstr="n", ansretry=True, ansdefault=None, timeout_sec=None):
    def timeout_interrupt(signum, frame):
        raise TimeoutError()

    assert not timeout_sec or ansdefault is not None, "Default answer need to be set when timeout is enabled"

    if ansretry is False:
        ansretry = 1
    elif ansretry is True:
        ansretry = float('inf')
    else:
        assert isinstance(ansretry, int)

    posstr = posstr.lower()
    negstr = negstr.lower()
    if ansdefault is not None:
        assert isinstance(ansdefault, bool)
        if ansdefault:
            ansdefault_str = posstr.lower()
            posstr = posstr.upper()
        else:
            ansdefault_str = negstr.lower()
            negstr = negstr.upper()
    else:
        assert ansretry == float('inf'), "No default answer for retry fallback"

    retry_count = 0
    while True:
        if timeout_sec:
            alarm_handler_ori = signal.signal(signal.SIGALRM, timeout_interrupt)
            signal.alarm(timeout_sec)
        try:
            ans = input(question + (" (timeout={}s)".format(timeout_sec) if timeout_sec else "") + " ({}|{}):".format(posstr, negstr))
        except TimeoutError:
            print()
            print("Answer timeout! using default answer: " + ansdefault_str)
            return ansdefault
        finally:
            if timeout_sec:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, alarm_handler_ori)

        retry_count += 1

        if ans.lower() == posstr.lower():
            return True
        elif ans.lower() == negstr.lower():
            return False
        elif ansdefault is not None and not ans:
            return ansdefault
        else:
            if retry_count < ansretry:
                print("Illegal answer! Retry")
                continue
            else:
                # not possible to reach here when ansdefault is None
                print("Illegal answer! Using default answer: " + ansdefault_str)
                return ansdefault

def get_args():
    argparser = argparse.ArgumentParser(prog="runman")
    subparsers = argparser.add_subparsers(help="Sub command")
    
    parser_list = subparsers.add_parser("list")
    mode_group = parser_list.add_mutually_exclusive_group()
    mode_group.add_argument("--archive", action="store_true")
    mode_group.add_argument("--trash", action="store_true")
    parser_list.set_defaults(func=list_runs)

    parser_remove = subparsers.add_parser("remove")
    mode_group = parser_remove.add_mutually_exclusive_group()
    mode_group.add_argument("--archive", action="store_true")
    mode_group.add_argument("--trash", action="store_true")
    parser_remove.add_argument("--permanant", action="store_true")
    parser_remove.add_argument("runid", action="append")
    parser_remove.set_defaults(func=remove_runs)

    parser_archive = subparsers.add_parser("archive")
    parser_archive.add_argument("--trash", action="store_true")
    parser_archive.add_argument("runid", action="append")
    parser_archive.set_defaults(func=archive_runs)

    parser_restore = subparsers.add_parser("restore")
    parser_restore.add_argument("--archive", action="store_true")
    parser_restore.add_argument("runid", action="append")
    parser_restore.set_defaults(func=restore_runs)

    parser_partaction = subparsers.add_parser("paction")
    parser_partaction.add_argument("--backup-dir", choices=["parchive", "archive", "trash"])
    parser_partaction.add_argument("--restore", action="store_true", default=None)
    parser_partaction.add_argument("--source-dir", choices=["parchive", "archive", "trash"])
    parser_partaction.add_argument("--dest-dir", choices=["parchive", "archive", "trash"])
    parser_partaction.add_argument("--type", choices=["checkpoint", "runs"], required=True)
    parser_partaction.add_argument("--runid-match-dest", action="store_true")
    parser_partaction.add_argument("--runid-pattern", action="append", help="regex", required=True)
    parser_partaction.add_argument("--runid-reverse", action="store_true")
    parser_partaction.add_argument("--part-pattern", action="append", help="regex")
    parser_partaction.add_argument("--part-reverse", action="store_true")
    parser_partaction_action_group = parser_partaction.add_mutually_exclusive_group()
    parser_partaction_action_group.add_argument("--print-source", action="store_true")
    parser_partaction_action_group.add_argument("--print-dest", action="store_true")
    parser_partaction_action_group.add_argument("--dry-run", action="store_true")
    parser_partaction_action_group.add_argument("--yes", action="store_true")
    parser_partaction.set_defaults(func=part_action)

    args = argparser.parse_args()
    return args

def _list_runs(root_dir):
    checkpoint_dir = os.path.join(root_dir, "checkpoint")
    run_dir = os.path.join(root_dir, "runs")

    empty_expid = list()
    all_runid = list()
    with os.scandir(checkpoint_dir) as entries_expid:
        for expid in entries_expid:
            is_exp_empty = True
            if expid.name.startswith('.') or expid.is_file():
                continue
            with os.scandir(os.path.join(checkpoint_dir, expid.name)) as entries_runid:
                for runid in entries_runid:
                    if runid.name.startswith('.') or runid.is_file():
                        continue
                    run_fullname = expid.name + "/" + runid.name
                    run_file_avail = os.path.isdir(os.path.join(run_dir, run_fullname))
                    print(run_fullname + ("" if run_file_avail else " (-run)"))
                    all_runid.append(run_fullname)
                    is_exp_empty = False
            if is_exp_empty:
                empty_expid.append(expid.name)

    if len(empty_expid) > 0:
        print("====== Empty expid in runs: " + str(empty_expid))

    empty_expid = list()
    with os.scandir(run_dir) as entries_expid:
        for expid in entries_expid:
            is_exp_empty = True
            if expid.name.startswith('.') or expid.is_file():
                continue
            with os.scandir(os.path.join(run_dir, expid.name)) as entries_runid:
                for runid in entries_runid:
                    if runid.name.startswith('.') or runid.is_file():
                        continue
                    is_exp_empty = False
                    run_fullname = expid.name + "/" + runid.name
                    if run_fullname in all_runid:
                        continue
                    print(run_fullname + " (-ckp)")
                    all_runid.append(run_fullname)
            if is_exp_empty:
                empty_expid.append(expid.name)
    
    if len(empty_expid) > 0:
        print("====== Empty expid in runs: " + str(empty_expid))
    
    if len(all_runid) == 0:
        print("No files found")

def list_runs(args):
    if args.archive:
        root_dir = os.path.join(ROOT_DIR, ARCHIVE_DIR)
    elif args.trash:
        root_dir = os.path.join(ROOT_DIR, TRASH_DIR)
    else:
        root_dir = ROOT_DIR
    if not os.path.isdir(root_dir):
        print("No files found")
    else:
        _list_runs(root_dir)

def _move_run(root_dir, expid, runid, dest_dir, checked=False, check_exp=True):
    exp_filename = os.path.join(root_dir, expid)
    run_filename = os.path.join(root_dir, expid, runid)
    if not checked:
        if not os.path.exists(run_filename):
            print("No files found")
            return

    if not os.path.isdir(dest_dir):
        if os.path.exists(dest_dir):
            print("Same expid but not dir already exists at " + dest_dir)
            return
        os.makedirs(dest_dir)
    elif os.path.exists(os.path.join(dest_dir, runid)):
        print("run (" + runid + ") already exists in " + dest_dir)
        return
    shutil.move(run_filename, dest_dir)

    if check_exp and len(os.listdir(exp_filename)) == 0 \
            and ask("The parent expid dir (" + exp_filename + ") is empty, do you want to delete it ?"):
        shutil.rmtree(exp_filename)

def _move_exp(root_dir, expid, dest_dir, checked=False):
    exp_filename = os.path.join(root_dir, expid)
    if not checked:
        if not os.path.exists(exp_filename):
            print("No files found")
            return

    if not os.path.isdir(dest_dir):
        if os.path.exists(dest_dir):
            print(dest_dir + " should be a dir")
            return
        os.makedirs(dest_dir)

    dest_exp_filename = os.path.join(dest_dir, expid)
    if os.path.exists(dest_exp_filename):
        print("exp (" + expid + ") already exists at " + dest_dir + ", trying to merge")
        with os.scandir(exp_filename) as entries_runid:
            for runid in entries_runid:
                dest_run_filename = os.path.join(dest_exp_filename, runid.name)
                if not os.path.exists(dest_run_filename):
                    shutil.move(os.path.join(exp_filename, runid.name), dest_exp_filename)
                else:
                    print("run (" + runid.name + ") already exists in " + dest_exp_filename)
                    return
        shutil.rmtree(exp_filename)
    else:
        shutil.move(exp_filename, dest_dir)

def _remove_run(root_dir, expid, runid, dest_subdir, permanant=False):
    exp_filename = os.path.join(root_dir, expid)
    run_filename = os.path.join(root_dir, expid, runid)
    if not os.path.exists(run_filename):
        print("No files found")
        return

    if not permanant:
        dest_dir = os.path.join(ROOT_DIR, TRASH_DIR, dest_subdir, expid)
        _move_run(root_dir, expid, runid, dest_dir, checked=True, check_exp=False)
    else:
        if ask("Are you sure to permanantly remove " + run_filename + " ?"):
            shutil.rmtree(run_filename)

    if len(os.listdir(exp_filename)) == 0 \
            and ask("The parent expid dir (" + exp_filename + ") is empty, do you want to delete it ?"):
        shutil.rmtree(exp_filename)

def _remove_exp(root_dir, expid, dest_subdir, permanant=False):
    exp_filename = os.path.join(root_dir, expid)
    if not os.path.exists(exp_filename):
        print("No files found")
        return

    if not permanant:
        dest_dir = os.path.join(ROOT_DIR, TRASH_DIR, dest_subdir)
        _move_exp(root_dir, expid, dest_dir, checked=True)
    else:
        if ask("Are you sure to permanantly remove " + exp_filename + " ?"):
            shutil.rmtree(exp_filename)

def remove_runs(args):
    if args.archive:
        root_dir = os.path.join(ROOT_DIR, ARCHIVE_DIR)
    elif args.trash:
        root_dir = os.path.join(ROOT_DIR, TRASH_DIR)
    else:
        root_dir = ROOT_DIR
    if not os.path.isdir(root_dir):
        print("No files found")
        return
    
    for runid in args.runid:
        runid = runid.strip().strip("/\\")
        run_hier = runid.split("/")

        if len(run_hier) == 2:
            _remove_run(os.path.join(root_dir, "checkpoint"), run_hier[0], run_hier[1], "checkpoint", permanant=args.trash or args.permanant)
            _remove_run(os.path.join(root_dir, "runs"), run_hier[0], run_hier[1], "runs", permanant=args.trash or args.permanant)
        elif len(run_hier) == 1:
            _remove_exp(os.path.join(root_dir, "checkpoint"), run_hier[0], "checkpoint", permanant=args.trash or args.permanant)
            _remove_exp(os.path.join(root_dir, "runs"), run_hier[0], "runs", permanant=args.trash or args.permanant)
        else:
            print("Invalid runid: " + runid)

def archive_runs(args):
    if args.trash:
        root_dir = os.path.join(ROOT_DIR, TRASH_DIR)
    else:
        root_dir = ROOT_DIR
    if not os.path.isdir(root_dir):
        print("No files found")
        return

    dest_dir = os.path.join(ROOT_DIR, ARCHIVE_DIR)
    
    for runid in args.runid:
        runid = runid.strip().strip("/\\")
        run_hier = runid.split("/")

        if len(run_hier) == 2:
            _move_run(os.path.join(root_dir, "checkpoint"), run_hier[0], run_hier[1], os.path.join(dest_dir, "checkpoint", run_hier[0]))
            _move_run(os.path.join(root_dir, "runs"), run_hier[0], run_hier[1], os.path.join(dest_dir, "runs", run_hier[0]))
        elif len(run_hier) == 1:
            _move_exp(os.path.join(root_dir, "checkpoint"), run_hier[0], os.path.join(dest_dir, "checkpoint"))
            _move_exp(os.path.join(root_dir, "runs"), run_hier[0], os.path.join(dest_dir, "runs"))
        else:
            print("Invalid runid: " + runid)

def restore_runs(args):
    if args.archive:
        root_dir = os.path.join(ROOT_DIR, ARCHIVE_DIR)
    else:
        root_dir = os.path.join(ROOT_DIR, TRASH_DIR)
    if not os.path.isdir(root_dir):
        print("No files found")
        return

    dest_dir = ROOT_DIR
    
    for runid in args.runid:
        runid = runid.strip().strip("/\\")
        run_hier = runid.split("/")

        if len(run_hier) == 2:
            _move_run(os.path.join(root_dir, "checkpoint"), run_hier[0], run_hier[1], os.path.join(dest_dir, "checkpoint", run_hier[0]))
            _move_run(os.path.join(root_dir, "runs"), run_hier[0], run_hier[1], os.path.join(dest_dir, "runs", run_hier[0]))
        elif len(run_hier) == 1:
            _move_exp(os.path.join(root_dir, "checkpoint"), run_hier[0], os.path.join(dest_dir, "checkpoint"))
            _move_exp(os.path.join(root_dir, "runs"), run_hier[0], os.path.join(dest_dir, "runs"))
        else:
            print("Invalid runid: " + runid)

def match_all(patterns, string, reverse_match):
    if patterns is not None:
        for pat in patterns:
            if re.search(pat, string) is None:
                return False if not reverse_match else True
    return True if not reverse_match else False

def _match_part(patterns, run_dir_inside, source_root, dest_root, reverse_match):
    matched_paths = []
    run_dir_source = os.path.join(source_root, run_dir_inside)
    if not os.path.isdir(run_dir_source):
        return matched_paths
    run_dir_dest = os.path.join(dest_root, run_dir_inside)
    for (dirpath_source, dirnames, filenames) in os.walk(run_dir_source):
        dirpath_rel = os.path.relpath(dirpath_source, run_dir_source)
        dirpath_dest = os.path.join(run_dir_dest, dirpath_rel)
        for subpath in filenames:
            full_path_dest = os.path.join(dirpath_dest, subpath)
            full_path_rel = os.path.normpath(os.path.join(dirpath_rel, subpath))
            if match_all(patterns, full_path_rel, reverse_match):
                if os.path.exists(full_path_dest):
                    raise RuntimeError("Dest path already exists '{}'".format(full_path_dest))
                matched_paths.append(full_path_rel)
        
        # can prune by deleting items
        dirnames_copy = dirnames.copy()
        found_counter = 0
        for idir, subpath in enumerate(dirnames_copy):
            full_path_dest = os.path.join(dirpath_dest, subpath)
            full_path_rel = os.path.normpath(os.path.join(dirpath_rel, subpath))
            if match_all(patterns, full_path_rel, reverse_match):
                if os.path.exists(full_path_dest):
                    if os.path.isdir(full_path_dest):
                        continue
                    else:
                        raise RuntimeError("Dest path already exists as a file '{}'".format(full_path_dest))
                del dirnames[idir - found_counter]
                found_counter += 1
                matched_paths.append(full_path_rel)
    return matched_paths

def part_action(args):
    is_tradition_srcdest = (args.backup_dir is not None or args.restore is not None) 
    is_new_srcdest = (args.source_dir is not None or args.dest_dir is not None)
    assert (is_tradition_srcdest and not is_new_srcdest) or \
        (not is_tradition_srcdest and is_new_srcdest) or \
        (not is_tradition_srcdest and not is_new_srcdest), "Valid option combination: (--backup-dir, --restore) or (--source-dir, --dest-dir)"

    dir_dict = dict(trash=TRASH_DIR, archive=ARCHIVE_DIR, parchive=PART_ARCHIVE_DIR)

    if is_tradition_srcdest or not is_new_srcdest:
        if args.backup_dir is None:
            args.backup_dir = "parchive"
        elif args.restore is None:
            args.restore = False

        backup_subdir = dir_dict[args.backup_dir]
        if args.restore:
            source_root = os.path.join(ROOT_DIR, backup_subdir, args.type)
            dest_root = os.path.join(ROOT_DIR, args.type)
        else:
            source_root = os.path.join(ROOT_DIR, args.type)
            dest_root = os.path.join(ROOT_DIR, backup_subdir, args.type)
    
    if is_new_srcdest:
        assert args.source_dir is not None and args.dest_dir is not None
        source_root = os.path.join(ROOT_DIR, dir_dict[args.source_dir], args.type)
        dest_root = os.path.join(ROOT_DIR, dir_dict[args.dest_dir], args.type)

    runid_match_root = dest_root if args.runid_match_dest else source_root
    if not os.path.isdir(runid_match_root):
        return
    ask_for_each = not (args.yes or args.dry_run or args.print_source or args.print_dest)
    take_action = not (args.dry_run or args.print_source or args.print_dest)
    involved_runids = list()
    with os.scandir(runid_match_root) as exp_dir_entries:
        for exp_dir in exp_dir_entries:
            if not exp_dir.is_dir():
                continue
            exp_dir_path = os.path.join(runid_match_root, exp_dir.name)
            if not os.path.isdir(exp_dir_path):
                continue
            with os.scandir(exp_dir_path) as run_dir_entries:
                for run_dir in run_dir_entries:
                    if not run_dir.is_dir():
                        continue
                    run_id_full = exp_dir.name + "/" + run_dir.name
                    # print(run_id_full)
                    if not match_all(args.runid_pattern, run_id_full, args.runid_reverse):
                        continue
                    run_dir_inside_path = os.path.join(exp_dir.name, run_dir.name)
                    is_run_matched = False
                    for match_path in _match_part(args.part_pattern, run_dir_inside_path, source_root, dest_root, args.part_reverse):
                        is_run_matched = True
                        inside_path = os.path.normpath(os.path.join(run_dir_inside_path, match_path))
                        source_path = os.path.join(source_root, inside_path)
                        source_path_base = os.path.dirname(source_path)
                        assert os.path.relpath(source_path, source_path_base) != "."
                        dest_path = os.path.join(dest_root, inside_path)
                        dest_path_base = os.path.dirname(dest_path)
                        assert os.path.relpath(dest_path, dest_path_base) != "."
                        if not ask_for_each or ask("Moving '{}' to '{}', confirm?".format(source_path, dest_path)):
                            if args.yes or args.dry_run:
                                print("Moving '{}' to '{}'".format(source_path, dest_path))
                            elif args.print_source:
                                print(source_path)
                            elif args.print_dest:
                                print(dest_path)
                            if take_action:
                                if not os.path.exists(dest_path_base):
                                    os.makedirs(dest_path_base)
                                shutil.move(source_path, dest_path_base)
                                
                                cur_base = os.path.abspath(source_path_base)
                                while len(os.listdir(cur_base)) == 0:
                                    os.rmdir(cur_base)
                                    cur_base = os.path.dirname(cur_base)
                                    if os.path.relpath(cur_base, source_root).startswith("."):
                                        break
                    if is_run_matched:
                        involved_runids.append(run_id_full)
    
    if args.dry_run:
        print("Involved runids: ")
        for runid in involved_runids:
            print("  " + runid)

if __name__ == "__main__":
    args = get_args()
    args.func(args)
