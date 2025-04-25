import { answer } from '@/models/answer';
import { intent } from '@/services/Intent/intent';
import { data } from '@/utils/data';
import { avatar } from '@/assets/logo.png';
import { message } from 'antd';
// import Service from '@/pages/QuanLyNguoiDung/service';
import { ActionType } from '@ant-design/pro-table';
import { useRef, useState } from 'react';
import fqas from '@/services/Fqas/fqas';
// import baseModel from './baseModel';

export interface IFqasRecord {
    id: string;
    question: string;
    chatbot_answer: string;
    human_answer: string;
    like: boolean;
    used: boolean;
}

export default () => {
    const [danhSach, setDanhSach] = useState<IFqasRecord[]>([]);
    const [showDrawer, setShowDrawer] = useState(false);
    const [edit, setEdit] = useState(false);
    const [loading, setLoading] = useState(false);
    const [record, setRecord] = useState<IFqasRecord | {}>({});
    const [visibleForm, setVisibleForm] = useState(false);
    const [total, setTotal] = useState(0);
    const [page, setPage] = useState(1);
    const [limit, setLimit] = useState(10);
    const [filterInfo, setFilterInfo] = useState<any>({});
    const [condition, setCondition] = useState<any>({});
    const [visible, setVisible] = useState<boolean>(false);

    const getData = async (IntentId: any) => {
        try {
            setLoading(true);
            const res = await fqas.get({
                "pagination[page]": page,
                "pagination[pageSize]": limit,
                ...condition,
            });
            // console.log(res);
            if (res.data) {
                setDanhSach(res.data?.map((item: any, index: number) => {
                    return {
                        id: item?.id,
                        question: item?.attributes?.question,
                        chatbot_answer: item?.attributes?.chatbot_answer,
                        human_answer: item?.attributes?.human_answer,
                        like: item?.attributes?.like,
                        used: item?.attributes?.used
                    }
                }));
                setTotal(res.meta?.pagination?.total);
            }
        } catch (error) {
            setDanhSach([]);
            setTotal(0);
            message.error('Lỗi thực hiện');
        } finally {
            setLoading(false);
        }
    };

    const getDataById = async (id: string) => {
        try {
            setLoading(true);
            const res = await fqas.get_by_id(id);
            console.log(res);
            if (res.status === 200) {
                setRecord(res.data);
            }
        } catch (error) {
            setRecord({});
            message.error('Lỗi thực hiện');
        } finally {
            setLoading(false);
        }
    };

    const add = async (data: any) => {
        try {
            setLoading(true);
            const res = await fqas.add(data);
            console.log(res);
            if (res.status === 200) {
                message.success('Thêm mới thành công');
                setVisibleForm(false);
            }
        } catch (error) {
            message.error('Lỗi thực hiện');
        } finally {
            setLoading(false);
        }
    };
    const addFile = async (data: any) => {
        try {
            setLoading(true);
            const res = await fqas.addFile(data);
            console.log(res);
            if (res.status === 200) {
                message.success('Thêm mới thành công');
                setVisibleForm(false);
            }
        } catch (error) {
            message.error('Lỗi thực hiện');
        } finally {
            setLoading(false);
        }
    };
    const addFileFqas = async (data: any) => {
        try {
            setLoading(true);
            const res = await fqas.addFileFqas(data);
            console.log(res);
            if (res.status === 200) {
                message.success('Thêm mới thành công');
                setVisibleForm(false);
            }
        } catch (error) {
            message.error('Lỗi thực hiện');
        } finally {
            setLoading(false);
        }
    };

    const upd = async (data: any) => {
        try {
            setLoading(true);
            const res = await fqas.upd(data);
            if (res.status === 200) {
                message.success('Cập nhật thành công');
                setVisibleForm(false);
            }
        } catch (error) {
            message.error('Lỗi thực hiện');
        } finally {
            setLoading(false);
        }
    };

    const del = async (id: string) => {
        try {
            setLoading(true);
            const res = await fqas.del(id);
            if (res.status === 200) {
                message.success('Xóa thành công');
            }
        } catch (error) {
            message.error('Lỗi thực hiện');
        } finally {
            setLoading(false);
        }
    };

    return {
        fqas,

        condition,
        danhSach,
        edit,
        filterInfo,
        loading,
        limit,
        total,
        page,
        record,
        visible,
        visibleForm,
        showDrawer,

        setCondition,
        setDanhSach,
        setEdit,
        setFilterInfo,
        setLimit,
        setTotal,
        setPage,
        setRecord,
        setVisible,
        setVisibleForm,
        setShowDrawer,

        getDataById,

        getData,
        add,
        upd,
        del,
        addFile,
        addFileFqas,
    };
};
