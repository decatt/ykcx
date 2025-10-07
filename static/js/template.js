// static/js/template.js

(function () {
  // 从模板注入数据
  const resultData = window.resultData;
  const downloadBtn = document.getElementById('downloadCsv');
  if (!downloadBtn || !Array.isArray(resultData)) return;

  downloadBtn.addEventListener('click', function () {
    const headers = ['IOL度数', '1m-矫正球镜度数 (D)'];

    function toCsvValue(value) {
      if (value === null || value === undefined) return '';
      const str = String(value);
      if (/[",\n]/.test(str)) {
        return '"' + str.replace(/"/g, '""') + '"';
      }
      return str;
    }

    const rows = resultData.map(row =>
      Array.isArray(row) ? row.map(toCsvValue).join(',') : ''
    );
    const csvContent = [headers.join(','), ...rows].join('\n');
    const blob = new Blob(['\ufeff' + csvContent], {
      type: 'text/csv;charset=utf-8;',
    });
    const url = URL.createObjectURL(blob);
    const tempLink = document.createElement('a');
    tempLink.href = url;
    tempLink.download = 'compute_result.csv';
    document.body.appendChild(tempLink);
    tempLink.click();
    document.body.removeChild(tempLink);
    URL.revokeObjectURL(url);
  });
})();

// ===== 一键填补 =====
(function () {
  const prefillBtn = document.getElementById('prefillBtn');
  const pidInput = document.getElementById('patient_id');
  const fieldNames = window.fieldNames;

  async function prefillByPid() {
    const pid = (pidInput?.value || '001').trim();
    if (!pid) return;

    prefillBtn.disabled = true;
    prefillBtn.textContent = '填充中…';

    try {
      const res = await fetch(`/api/prefill?pid=${encodeURIComponent(pid)}`);
      const data = await res.json();
      if (!data.ok || !Array.isArray(data.values)) {
        alert('预填失败：' + (data.error || '未知错误'));
        return;
      }

      data.values.forEach((v, i) => {
        const name = fieldNames[i];
        if (!name) return;
        const input = document.getElementById(name);
        if (input) {
          input.value = v;
          input.dispatchEvent(new Event('input', { bubbles: true }));
          input.dispatchEvent(new Event('change', { bubbles: true }));
        }
      });
    } catch (err) {
      alert('网络或服务错误：' + err.message);
    } finally {
      prefillBtn.disabled = false;
      prefillBtn.textContent = '一键填补';
    }
  }

  if (prefillBtn) prefillBtn.addEventListener('click', prefillByPid);
})();

// ===== 选择度数 =====
(function () {
  const dropdown = document.getElementById('resultDropdown');
  const selectBtn = document.getElementById('selectDegreeBtn');

  if (!dropdown || !selectBtn) return;

  selectBtn.addEventListener('click', async function () {
    const option = dropdown.selectedOptions[0];
    if (!option || !option.dataset.iol) {
      alert('请选择一个度数');
      return;
    }

    const payload = {
      iol_power: option.dataset.iol ? parseFloat(option.dataset.iol) : null,
      sphere: option.dataset.sphere ? parseFloat(option.dataset.sphere) : null,
    };

    try {
      const response = await fetch('/api/select_iol', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await response.json();
      if (!data.ok) throw new Error(data.error || '未知错误');

      alert(data.message || '已提交选择');
    } catch (error) {
      alert('提交失败：' + error.message);
    }
  });
})();
