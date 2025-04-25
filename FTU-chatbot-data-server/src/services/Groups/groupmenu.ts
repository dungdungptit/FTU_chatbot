/* eslint-disable no-param-reassign */
/* eslint-disable no-underscore-dangle */
/* eslint-disable @typescript-eslint/naming-convention */
import axios from '@/utils/axios';
import { ip } from '@/utils/ip';


export interface IPayload {
  page: number,
  limit: number,
  cond: object,
}

class Groups<T> {
  name: string;
  url: string;

  constructor({ name, url }: { name: string; url: string }) {
    this.name = name;
    this.url = url || name;
  }

  del = async (payload: any) => {
    return axios.delete(`${ip}/group_menu_by_id`, { params: payload });
  };

  get = async (payload: IPayload) => {
    // console.log(payload);
    return axios.get(`${ip}/${this.url}`, { params: payload });
  };
  getDataGroupMenu = async (payload: IPayload) => {
    // console.log(payload);
    return axios.get(`${ip}/group_menu`, { params: payload });
  };

  add = async (payload: T) => {
    Object.keys(payload).forEach((key) => {
      // payload[key] = payload[key];
      payload[key] = payload[key];
    });
    const formData = new FormData();
    Object.keys(payload).forEach((key) => {
      formData.append(key, payload[key]);
    });
    return axios.post(`${ip}/group_menu_by_id`, formData);
  };

  add2 = async (payload: T) => {
    Object.keys(payload).forEach((key) => {
      // payload[key] = payload[key];
      payload[key] = payload[key];
    });
    return axios.post(`${ip}/${this.url}/`, payload);
  };

  upd = async (payload: T & { id: string | undefined }) => {
    const { id } = payload;
    payload.id = undefined;
    Object.keys(payload).forEach((key) => {
      payload[key] = payload[key];
    });
    const formData = new FormData();
    Object.keys(payload).forEach((key) => {
      formData.append(key, payload[key]);
    });
    formData.delete('id');
    formData.append('id', id || '');
    return axios.put(`${ip}/group_menu`, formData);
  };
}

const groups = new Groups({ name: 'groups', url: 'group' });

export default groups;
