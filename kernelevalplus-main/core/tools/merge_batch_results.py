#!/usr/bin/env python3
"""
合并所有批量实验结果到指定目录
"""

import json
from pathlib import Path
from datetime import datetime

def collect_results_from_batch_dir(batch_dir: Path) -> list:
    """从批量目录收集结果"""
    results = []

    # 检查是否有汇总文件
    summary_file = batch_dir / "batch_summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
            results.extend(summary.get("results", []))
        return results

    # 如果没有汇总文件，从子目录收集
    for subdir in batch_dir.iterdir():
        if not subdir.is_dir():
            continue

        # 查找 variant 子目录
        variant_dirs = list(subdir.glob("*/test_results.json"))
        for variant_dir in variant_dirs:
            test_result_file = variant_dir
            try:
                with open(test_result_file) as f:
                    test_result = json.load(f)

                # 构造结果条目
                attempt_id = subdir.name
                variant = test_result_file.parent.name

                entry = {
                    "definition": f"{attempt_id}.json",
                    "provider": "unknown",
                    "model": "unknown",
                    "success": test_result.get("correctness", {}).get("passed", False),
                    "attempt_id": attempt_id,
                    "test_result": test_result
                }
                results.append(entry)
            except Exception as e:
                print(f"  ⚠️  无法读取 {test_result_file}: {e}")

    return results

def main():
    project_root = Path(__file__).parent.parent.parent
    generated_dir = project_root / "llm_kernel_test" / "sandbox" / "generated"

    # 目标目录
    target_dir = generated_dir / "deepseek-v3.2-20260206-225337"

    # 源目录
    source_dirs = [
        generated_dir / "batch_20260206_132409_unified_v2",
        generated_dir / "deepseek-v3.2-20260206-223743",
        generated_dir / "deepseek-v3.2-20260206-224713",
    ]

    print("=" * 60)
    print("合并批量实验结果")
    print("=" * 60)
    print(f"目标目录: {target_dir}")
    print()

    # 收集所有结果
    all_results = []
    seen_attempt_ids = set()

    # 先读取目标目录中已有的结果
    target_summary = target_dir / "batch_summary.json"
    if target_summary.exists():
        with open(target_summary) as f:
            target_data = json.load(f)
            existing_results = target_data.get("results", [])
            for r in existing_results:
                attempt_id = r.get("attempt_id")
                if attempt_id:
                    seen_attempt_ids.add(attempt_id)
            all_results.extend(existing_results)
        print(f"📋 目标目录已有 {len(existing_results)} 条结果")

    # 从源目录收集结果
    for source_dir in source_dirs:
        if not source_dir.exists():
            print(f"⚠️  跳过不存在的目录: {source_dir}")
            continue

        print(f"\n📂 处理: {source_dir.name}")
        new_results = collect_results_from_batch_dir(source_dir)

        # 去重
        added = 0
        for r in new_results:
            attempt_id = r.get("attempt_id")
            if attempt_id and attempt_id not in seen_attempt_ids:
                all_results.append(r)
                seen_attempt_ids.add(attempt_id)
                added += 1

        print(f"   找到 {len(new_results)} 条，新增 {added} 条（去重后）")

    # 统计
    total = len(all_results)
    success = sum(1 for r in all_results if r.get("success"))
    failed = total - success

    print(f"\n{'=' * 60}")
    print(f"合并统计")
    print(f"{'=' * 60}")
    print(f"总计: {total}")
    print(f"成功: {success} ({success*100//total if total > 0 else 0}%)")
    print(f"失败: {failed}")

    # 保存结果
    summary = {
        "batch_name": "deepseek-v3.2-20260206-225337",
        "timestamp": datetime.now().isoformat(),
        "merged_from": [str(d) for d in source_dirs if d.exists()],
        "total": total,
        "success": success,
        "failed": failed,
        "results": all_results
    }

    summary_file = target_dir / "batch_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n💾 汇总结果已保存: {summary_file}")

    # 保存 CSV
    csv_file = target_dir / "batch_results.csv"
    with open(csv_file, 'w') as f:
        f.write("definition,attempt_id,success,compilation,correctness\n")
        for r in all_results:
            def_name = Path(r.get("definition", "")).name
            attempt_id = r.get("attempt_id", "")
            success_mark = "✓" if r.get("success") else "✗"
            comp = "✓" if r.get("test_result", {}).get("compilation", {}).get("success") else "✗"
            corr = "✓" if r.get("test_result", {}).get("correctness", {}).get("passed") else "✗"
            f.write(f"{def_name},{attempt_id},{success_mark},{comp},{corr}\n")

    print(f"💾 CSV 结果已保存: {csv_file}")

if __name__ == "__main__":
    main()
