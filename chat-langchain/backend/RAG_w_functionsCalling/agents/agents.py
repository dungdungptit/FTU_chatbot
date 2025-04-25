from RAG_w_functionsCalling.agents.SearchAgent import SearchAgent
from RAG_w_functionsCalling.agents.WelcomeAgent import WelcomeAgent
from RAG_w_functionsCalling.agents.ChiTieuTuyenSinhAgent import ChiTieuTuyenSinhAgent
from RAG_w_functionsCalling.agents.ChinhSachMoiAgent import ChinhSachMoiAgent
from RAG_w_functionsCalling.agents.ChinhSachUuTienAgent import ChinhSachUuTienAgent
from RAG_w_functionsCalling.agents.DiemTrungTuyenAgent import DiemTrungTuyenAgent
from RAG_w_functionsCalling.agents.DieuKienXetTuyenAgent import DieuKienXetTuyenAgent
from RAG_w_functionsCalling.agents.HocBongAgent import HocBongAgent
from RAG_w_functionsCalling.agents.KyTucXaAgent import KyTucXaAgent
from RAG_w_functionsCalling.agents.LePhiAgent import LePhiAgent
from RAG_w_functionsCalling.agents.NganhHocAgent import NganhHocAgent
from RAG_w_functionsCalling.agents.PhuongThucXetTuyenAgent import PhuongThucXetTuyenAgent
from RAG_w_functionsCalling.agents.QuyDoiChungChiAgent import QuyDoiChungChiAgent
from RAG_w_functionsCalling.agents.TuyenSinhAgent import TuyenSinhAgent


from RAG_w_functionsCalling.core.models import get_chat_completion
from chain_code import normalize_replace_abbreviation_text

functions = {
    "Search": SearchAgent.search_agent,
    "Welcome": WelcomeAgent.welcome_agent,
    "ChiTieuTuyenSinh": ChiTieuTuyenSinhAgent.chi_tieu_tuyen_sinh_agent,
    "ChinhSachMoi": ChinhSachMoiAgent.chinh_sach_moi_agent,
    "ChinhSachUuTien": ChinhSachUuTienAgent.chinh_sach_uu_tien_agent,
    "DiemTrungTuyen": DiemTrungTuyenAgent.diem_trung_tuyen_agent,
    "DieuKienXetTuyen": DieuKienXetTuyenAgent.dieu_kien_xet_tuyen_agent,
    "HocBong": HocBongAgent.hoc_bong_agent,
    "KyTucXa": KyTucXaAgent.ky_tuc_xa_agent,
    "LePhi": LePhiAgent.le_phi_agent,
    "NganhVaChuyenNganh": NganhHocAgent.nganh_va_chuyen_nganh_agent,
    "PhuongThucXetTuyen": PhuongThucXetTuyenAgent.phuong_thuc_xet_tuyen_agent,
    "QuyDoiChungChi": QuyDoiChungChiAgent.quy_doi_chung_chi_agent,
    "TuyenSinh": TuyenSinhAgent.tuyen_sinh_agent,
}

functions_description = "\n".join(
    [f"{name}: {obj.description}" for name, obj in functions.items()]
)
